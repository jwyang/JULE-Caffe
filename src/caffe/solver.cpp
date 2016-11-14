#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>
#include "hdf5/hdf5.h"
#include "hdf5/hdf5_hl.h"

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/computeMergeLoss.hpp"
#include "caffe/util/updateAc2merged.hpp"

#define use_gpu_update_labels 1

namespace caffe {

	template<typename Dtype>
	void Solver<Dtype>::SetActionFunction(ActionCallback func) {
		action_request_function_ = func;
	}

	template<typename Dtype>
	SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
		if (action_request_function_) {
			// If the external request function has been set, call it.
			return action_request_function_();
		}
		return SolverAction::NONE;
	}

	template <typename Dtype>
	Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
		: net_(), callbacks_(), root_solver_(root_solver),
		requested_early_exit_(false) {
		Init(param);
	}

	template <typename Dtype>
	Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
		: net_(), callbacks_(), root_solver_(root_solver),
		requested_early_exit_(false) {
		SolverParameter param;
		ReadProtoFromTextFileOrDie(param_file, &param);
		Init(param);
	}

	template <typename Dtype>
	void Solver<Dtype>::Init(const SolverParameter& param) {
		CHECK(Caffe::root_solver() || root_solver_)
			<< "root_solver_ needs to be set for all non-root solvers";
		LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
			<< std::endl << param.DebugString();
		param_ = param;
		CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
		if (Caffe::root_solver() && param_.random_seed() >= 0) {
			Caffe::set_random_seed(param_.random_seed());
		}
		// Scaffolding code
		InitTrainNet();
		if (Caffe::root_solver()) {
			InitTestNets();
			LOG(INFO) << "Solver scaffolding done.";
		}
		iter_ = 0;
		current_step_ = 0;
		ac_lcc_ = false;
	}

	template <typename Dtype>
	void Solver<Dtype>::InitTrainNet() {
		const int num_train_nets = param_.has_net() + param_.has_net_param() +
			param_.has_train_net() + param_.has_train_net_param();
		const string& field_names = "net, net_param, train_net, train_net_param";
		CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
			<< "using one of these fields: " << field_names;
		CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
			<< "one of these fields specifying a train_net: " << field_names;
		NetParameter net_param;
		if (param_.has_train_net_param()) {
			LOG_IF(INFO, Caffe::root_solver())
				<< "Creating training net specified in train_net_param.";
			net_param.CopyFrom(param_.train_net_param());
		}
		else if (param_.has_train_net()) {
			LOG_IF(INFO, Caffe::root_solver())
				<< "Creating training net from train_net file: " << param_.train_net();
			ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
		}
		if (param_.has_net_param()) {
			LOG_IF(INFO, Caffe::root_solver())
				<< "Creating training net specified in net_param.";
			net_param.CopyFrom(param_.net_param());
		}
		if (param_.has_net()) {
			LOG_IF(INFO, Caffe::root_solver())
				<< "Creating training net from net file: " << param_.net();
			ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
		}
		// Set the correct NetState.  We start with the solver defaults (lowest
		// precedence); then, merge in any NetState specified by the net_param itself;
		// finally, merge in any NetState specified by the train_state (highest
		// precedence).
		NetState net_state;
		net_state.set_phase(TRAIN);
		net_state.MergeFrom(net_param.state());
		net_state.MergeFrom(param_.train_state());
		net_param.mutable_state()->CopyFrom(net_state);
		if (Caffe::root_solver()) {
			net_.reset(new Net<Dtype>(net_param));
		}
		else {
			net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
		}
	}

	template <typename Dtype>
	void Solver<Dtype>::InitTestNets() {
		CHECK(Caffe::root_solver());
		const bool has_net_param = param_.has_net_param();
		const bool has_net_file = param_.has_net();
		const int num_generic_nets = has_net_param + has_net_file;
		CHECK_LE(num_generic_nets, 1)
			<< "Both net_param and net_file may not be specified.";
		const int num_test_net_params = param_.test_net_param_size();
		const int num_test_net_files = param_.test_net_size();
		const int num_test_nets = num_test_net_params + num_test_net_files;
		if (num_generic_nets) {
			CHECK_GE(param_.test_iter_size(), num_test_nets)
				<< "test_iter must be specified for each test network.";
		}
		else {
			CHECK_EQ(param_.test_iter_size(), num_test_nets)
				<< "test_iter must be specified for each test network.";
		}
		// If we have a generic net (specified by net or net_param, rather than
		// test_net or test_net_param), we may have an unlimited number of actual
		// test networks -- the actual number is given by the number of remaining
		// test_iters after any test nets specified by test_net_param and/or test_net
		// are evaluated.
		const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
		const int num_test_net_instances = num_test_nets + num_generic_net_instances;
		if (param_.test_state_size()) {
			CHECK_EQ(param_.test_state_size(), num_test_net_instances)
				<< "test_state must be unspecified or specified once per test net.";
		}
		if (num_test_net_instances) {
			CHECK_GT(param_.test_interval(), 0);
		}
		int test_net_id = 0;
		vector<string> sources(num_test_net_instances);
		vector<NetParameter> net_params(num_test_net_instances);
		for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
			sources[test_net_id] = "test_net_param";
			net_params[test_net_id].CopyFrom(param_.test_net_param(i));
		}
		for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
			sources[test_net_id] = "test_net file: " + param_.test_net(i);
			ReadNetParamsFromTextFileOrDie(param_.test_net(i),
				&net_params[test_net_id]);
		}
		const int remaining_test_nets = param_.test_iter_size() - test_net_id;
		if (has_net_param) {
			for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
				sources[test_net_id] = "net_param";
				net_params[test_net_id].CopyFrom(param_.net_param());
			}
		}
		if (has_net_file) {
			for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
				sources[test_net_id] = "net file: " + param_.net();
				ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
			}
		}
		test_nets_.resize(num_test_net_instances);
		for (int i = 0; i < num_test_net_instances; ++i) {
			// Set the correct NetState.  We start with the solver defaults (lowest
			// precedence); then, merge in any NetState specified by the net_param
			// itself; finally, merge in any NetState specified by the test_state
			// (highest precedence).
			NetState net_state;
			net_state.set_phase(TEST);
			net_state.MergeFrom(net_params[i].state());
			if (param_.test_state_size()) {
				net_state.MergeFrom(param_.test_state(i));
			}
			net_params[i].mutable_state()->CopyFrom(net_state);
			LOG(INFO)
				<< "Creating test net (#" << i << ") specified by " << sources[i];
			if (Caffe::root_solver()) {
				test_nets_[i].reset(new Net<Dtype>(net_params[i]));
			}
			else {
				test_nets_[i].reset(new Net<Dtype>(net_params[i],
					root_solver_->test_nets_[i].get()));
			}
			test_nets_[i]->set_debug_info(param_.debug_info());
		}
	}

	template <typename Dtype>
	void Solver<Dtype>::Step(int iters) {
		vector<Blob<Dtype>*> bottom_vec;
		const int start_iter = iter_;
		const int stop_iter = iter_ + iters;
		int average_loss = this->param_.average_loss();
		vector<Dtype> losses;
		Dtype smoothed_loss = 0;

		while (iter_ < stop_iter) {
			// zero-init the params
			net_->ClearParamDiffs();
			if (param_.test_interval() && iter_ % param_.test_interval() == 0
				&& (iter_ > 0 || param_.test_initialization())
				&& Caffe::root_solver()) {
				TestAll();
				if (requested_early_exit_) {
					// Break out of the while loop because stop was requested while testing.
					break;
				}
			}

			for (int i = 0; i < callbacks_.size(); ++i) {
				callbacks_[i]->on_start();
			}
			const bool display = param_.display() && iter_ % param_.display() == 0;
			net_->set_debug_info(display && param_.debug_info());
			// accumulate the loss and gradient
			Dtype loss = 0;
			for (int i = 0; i < param_.iter_size(); ++i) {
				loss += net_->ForwardBackward(bottom_vec);
			}
			loss /= param_.iter_size();
			// average the loss across iterations for smoothed reporting
			if (losses.size() < average_loss) {
				losses.push_back(loss);
				int size = losses.size();
				smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
			}
			else {
				int idx = (iter_ - start_iter) % average_loss;
				smoothed_loss += (loss - losses[idx]) / average_loss;
				losses[idx] = loss;
			}
			if (display) {
				LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
					<< ", loss = " << smoothed_loss;
				const vector<Blob<Dtype>*>& result = net_->output_blobs();
				int score_index = 0;
				for (int j = 0; j < result.size(); ++j) {
					const Dtype* result_vec = result[j]->cpu_data();
					const string& output_name =
						net_->blob_names()[net_->output_blob_indices()[j]];
					const Dtype loss_weight =
						net_->blob_loss_weights()[net_->output_blob_indices()[j]];
					for (int k = 0; k < result[j]->count(); ++k) {
						ostringstream loss_msg_stream;
						if (loss_weight) {
							loss_msg_stream << " (* " << loss_weight
								<< " = " << loss_weight * result_vec[k] << " loss)";
						}
						LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
							<< score_index++ << ": " << output_name << " = "
							<< result_vec[k] << loss_msg_stream.str();
					}
				}
			}
			for (int i = 0; i < callbacks_.size(); ++i) {
				callbacks_[i]->on_gradients_ready();
			}
			ApplyUpdate();

			// Increment the internal iter_ counter -- its value should always indicate
			// the number of times the weights have been updated.
			++iter_;

			SolverAction::Enum request = GetRequestedAction();

			// Save a snapshot if needed.
			if ((param_.snapshot()
				&& iter_ % param_.snapshot() == 0
				&& Caffe::root_solver()) ||
				(request == SolverAction::SNAPSHOT)) {
				Snapshot();
			}
			if (SolverAction::STOP == request) {
				requested_early_exit_ = true;
				// Break out of training loop.
				break;
			}
		}
	}

	template <typename Dtype>
	void Solver<Dtype>::Solve(const char* resume_file) {
		CHECK(Caffe::root_solver());
		LOG(INFO) << "Solving " << net_->name();
		LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

		// Initialize to false every time we start solving.
		requested_early_exit_ = false;

		if (resume_file) {
			LOG(INFO) << "Restoring previous solver status from " << resume_file;
			Restore(resume_file);
		}

		// Set data labels priorly  
		// net_->SetDataLabels(NULL, 0, 0);

		// For a network that is trained by the solver, no bottom or top vecs
		// should be given, and we will just provide dummy vecs.
		Step(param_.max_iter() - iter_);
		// If we haven't already, save a snapshot after optimization, unless
		// overridden by setting snapshot_after_train := false
		if (param_.snapshot_after_train()
			&& (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
			Snapshot();
		}
		if (requested_early_exit_) {
			LOG(INFO) << "Optimization stopped early.";
			return;
		}
		// After the optimization is done, run an additional train and test pass to
		// display the train and test loss/outputs if appropriate (based on the
		// display and test_interval settings, respectively).  Unlike in the rest of
		// training, for the train net we only run a forward pass as we've already
		// updated the parameters "max_iter" times -- this final pass is only done to
		// display the loss, which is computed in the forward pass.
		if (param_.display() && iter_ % param_.display() == 0) {
			Dtype loss;
			net_->ForwardPrefilled(&loss);
			LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
		}
		if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
			TestAll();
		}
		LOG(INFO) << "Optimization Done.";
	}

	// added by Jianwei Yang @09/28/2015
	// main body for extracting features given model parameters
	template <typename Dtype>
	float* Solver<Dtype>::ExtFeatures(int idx_start, int idx_end) {
		assert(idx_start >= 0 && idx_start < num_);
		assert(idx_end >= 0 && idx_end < num_);
		assert(idx_start <= idx_end);

		string name_layer4feat = name_layer4feat_;
		if ((epoch_ == 0 && !is_final_eval_) || ac_lcc_)
			name_layer4feat = "data";
		const shared_ptr<Net<Dtype> >& test_net = test_nets_[0];

		int pos_cur = test_net->GetDataPos();
		long long int num_mini_batches = std::ceil(double(num_) / double(mini_batch_size_test_));
		int image_index = 0;
		vector<Blob<Dtype>*> input_vec;
		
		const boost::shared_ptr<Blob<Dtype> > feature_blob = net_->blob_by_name(name_layer4feat);
		dim_feature_ = feature_blob->count() / feature_blob->num();
		LOG(INFO) << "Feature Dimension: " << dim_feature_;
		// allocate memory for features
		float* features_all = new float[num_mini_batches * mini_batch_size_test_ * dim_feature_];
		for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) { // forward all minibatches
			test_net->Forward(input_vec);
			const boost::shared_ptr<Blob<Dtype> > feature_blob = test_net->blob_by_name(name_layer4feat);
			int batch_size = feature_blob->num();
			int dim_features = feature_blob->count() / batch_size;
			const Dtype* feature_blob_data;
			for (int n = 0; n < batch_size; ++n) {
				feature_blob_data = feature_blob->cpu_data() +
					feature_blob->offset(n);
				memcpy(features_all + batch_index * batch_size * dim_feature_ + n * dim_feature_,
					feature_blob_data, dim_feature_ * sizeof(Dtype));
				++image_index;
				if (image_index % 1000 == 0) {
					LOG(ERROR) << "Extracted features of " << image_index <<
						" query images for feature blob " << name_layer4feat;
				}
			}  // for (int n = 0; n < batch_size; ++n)
		}  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

		// align 
		float* features = new float[(idx_end - idx_start + 1) * dim_feature_];
		if (idx_start >= pos_cur) {
			memcpy(features, features_all + (idx_start - pos_cur) * dim_feature_,
				(idx_end - idx_start + 1) * dim_feature_ * sizeof(float));
		}
		else if (idx_start < pos_cur && idx_end >= pos_cur){
			memcpy(features + (pos_cur - idx_start) * dim_feature_, features_all, (idx_end - pos_cur + 1) * dim_feature_ * sizeof(float));
			memcpy(features, features_all + (num_ - pos_cur + idx_start) * dim_feature_,
				(pos_cur - idx_start) * dim_feature_ * sizeof(float));
		}
		else if (idx_end < pos_cur) {
			memcpy(features, features_all + (num_ - pos_cur + idx_start) * dim_feature_,
				(idx_end - idx_start) * dim_feature_ * sizeof(float));
		}

		delete[]features_all;

		return features;
	}

	// added by Jianwei Yang @ 09/28/2015
	template <typename Dtype>
	void Solver<Dtype>::MergeClusters(vector<vector<int64> >& label_indice, cv::Mat& W_samples) {
		int64 num_elements = int64(W_samples.rows) * int64(W_samples.cols);
		float* W_samples_vec = new float[num_elements];
		for (int64 i = 0; i < W_samples.rows; ++i) {
			for (int64 j = 0; j < W_samples.cols; ++j) {
				W_samples_vec[i * W_samples.cols + j] = W_samples.at<float>(i, j);
			}
		}

		int num_samples = W_samples.rows;
		// compute intra_affinity, A(C) = \sum_m \sum_n W_m_n * W_n_m, m, n \in C
		vector<float> affinity_intra(label_indice.size());
		for (int i = 0; i < label_indice.size(); ++i) {
			float val = 0;
			for (int m = 0; m < label_indice[i].size(); ++m) {
				for (int n = 0; n < label_indice[i].size(); ++n) {
					if (m == n)
						continue;
					float entry_m_n = W_samples.at<float>(label_indice[i][m], label_indice[i][n]);
					float entry_n_m = W_samples.at<float>(label_indice[i][n], label_indice[i][m]);
					val += entry_m_n * entry_n_m;
				}
			}

			affinity_intra[i] = val;
		}

		// compute inter_affinity = A(C_i -> C_j)
		cv::Mat affinity_inter = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
		cv::Mat affinity_inter_mlink = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);

#pragma omp parallel for
		for (int i = 0; i < label_indice.size(); ++i) {
			for (int j = 0; j < label_indice.size(); ++j) {
				if (i == j) {
					affinity_inter.at<float>(j, i) = 0;
					affinity_inter_mlink.at<float>(i, j) = 0;
					affinity_inter_mlink.at<float>(j, i) = 0;
					continue;
				}
				float A_c_i_j = 0;
				float W_max_i_j = 0;
				float W_max_j_i = 0;
				for (int m = 0; m < label_indice[i].size(); ++m) {

					float s_W_c_j_i = 0;
					for (int n = 0; n < label_indice[j].size(); ++n) {
						s_W_c_j_i += W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
						W_max_i_j = max(W_max_i_j, W_samples.at<float>(label_indice[j][n], label_indice[i][m]));
					}

					float s_W_c_i_j = 0;
					for (int n = 0; n < label_indice[j].size(); ++n) {
						s_W_c_i_j += W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
						W_max_j_i = max(W_max_j_i, W_samples.at<float>(label_indice[i][m], label_indice[j][n]));
					}

					A_c_i_j += s_W_c_j_i * s_W_c_i_j;
				}

				affinity_inter_mlink.at<float>(i, j) = W_max_i_j + W_max_j_i;
				affinity_inter_mlink.at<float>(j, i) = W_max_i_j + W_max_j_i;
				//int64 msize = int64(label_indice[i].size()) * int64(label_indice[j].size());
				//float* W_samples_sub_j_i = new float[msize];
				//float* W_samples_sub_i_j = new float[msize];
				//for (int m = 0; m < label_indice[j].size(); ++m) {
				//	for (int n = 0; n < label_indice[i].size(); ++n) {
				//		W_samples_sub_j_i[m * label_indice[i].size() + n] = W_samples_vec[label_indice[j][m] * W_samples.cols
				//			+ label_indice[i][n]];
				//		W_samples_sub_i_j[n * label_indice[j].size() + m] = W_samples_vec[label_indice[i][n] * W_samples.cols
				//			+ label_indice[j][m]];
				//	}
				//}
				//int64 msize_product = int64(label_indice[j].size()) * int64(label_indice[j].size());
				//float* W_samples_product = new float[msize_product];
				//caffe::caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, label_indice[j].size(), label_indice[j].size(), label_indice[i].size(),
				//	1.0f, W_samples_sub_j_i, W_samples_sub_i_j, 0.0f, W_samples_product);
				//A_c_i_j = caffe::caffe_cpu_asum(msize_product, W_samples_product);

				////affinity_inter_i_merged = computeAc2merged(W_samples_sub_j_i, W_samples_sub_i_j,
				////	label_indice[j].size(), label_indice[i].size());

				//delete[]W_samples_sub_j_i;
				//delete[]W_samples_sub_i_j;
				//delete[]W_samples_product;

				//for (int m = 0; m < label_indice[i].size(); ++m) {
				//	for (int n = 0; n < label_indice[j].size(); ++n) {
				//		float entry_m_n = W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
				//		float entry_n_m = W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
				//		A_c_i_j += entry_m_n * entry_n_m;
				//	}
				//}

				affinity_inter.at<float>(j, i) = A_c_i_j;
			}
		}

		// compute inter_affinity using GPU


		// find optimal cluster pair so that the loss decrease mostly.
		// we are not going to traverse all possible pairs because it
		// is exhausted if there are thousands of clusters, alternatively,
		// we find top-K candidates with largest inter_affinity
		float* affinity_inter_sym_vec = new float[int64(label_indice.size()) * int64(label_indice.size())];
		cv::Mat affinity_inter_sym = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
		cv::Mat affinity_inter_T;
		cv::transpose(affinity_inter, affinity_inter_T);

		for (int64 i = 0; i < affinity_inter_sym.rows; ++i) {
			for (int64 j = 0; j < affinity_inter_sym.cols; ++j) {
				if (j <= i) {
					affinity_inter_sym.at<float>(i, j) = 0;
					affinity_inter_sym_vec[i * affinity_inter_sym.cols + j] = 0;
				}
				else {
					affinity_inter_sym.at<float>(i, j) 
						= affinity_inter.at<float>(i, j) / label_indice[i].size() / label_indice[i].size()
						+ affinity_inter_T.at<float>(i, j) / label_indice[j].size() / label_indice[j].size();
					affinity_inter_sym_vec[i * affinity_inter_sym.cols + j] = affinity_inter_sym.at<float>(i, j);

				}
			}
		}

		int test_iter = 1000;
		int iter = 0;
		int num_categories = label_indice.size();
		int* updated_labels = new int[num_samples];

		float loss_intra_new, loss_inter_new;
		cv::Mat affinity_inter_sym_temp;
		
		double minVal, maxVal;
		int minLoc[2], maxLoc[2];
		vector<float> deltas(num_categories);
		vector<int> maxLocs(num_categories);

		cv::Mat decm(1, num_categories, CV_32FC1);
		for (int i = 0; i < num_categories; ++i) {
			decm.at<float>(i) = 1.0f / label_indice[i].size() / label_indice[i].size();
		}

		// merge clusters
		cv::Mat affinity_inter_sum_row, affinity_inter_sum_col;
		cv::reduce(affinity_inter, affinity_inter_sum_row, 0, CV_REDUCE_SUM);
		cv::reduce(affinity_inter, affinity_inter_sum_col, 1, CV_REDUCE_SUM);
		
		//for (int i = 0; i < affinity_inter.cols; ++i) {
		//	if (affinity_inter_sum_row.at<float>(i) == 0 || affinity_inter_sum_col.at<float>(i) == 0) {
		//		int idx_a, idx_b;
		//		cv::Point maxLoc, minLoc;
		//		float maxVal, minVal;
		//		cv::minMaxLoc(affinity_inter_mlink.row(i), NULL, NULL, &minLoc, &maxLoc);
		//		idx_a = min(i, maxLoc.x);
		//		idx_b = max(i, maxLoc.x);
		//		// update label_indice, merge idx_b to idx_a, and then clear idx_b
		//		float rat = float(label_indice[idx_a].size()) / float(label_indice[idx_a].size() + label_indice[idx_b].size());
		//		label_indice[idx_a].insert(label_indice[idx_a].end(), label_indice[idx_b].begin(), label_indice[idx_b].end());
		//		label_indice[idx_b].clear();
		//		decm.at<float>(idx_b) = 0;
		//		decm.at<float>(idx_a) = 1.0f / label_indice[idx_a].size() / label_indice[idx_a].size();

		//		//// update the intra affinity and inter affinity

		//		/// update intra affinity
		//		//affinity_intra[idx_a] = (affinity_intra[idx_a] + affinity_intra[idx_b] +
		//		//	affinity_inter.at<float>(idx_a, idx_b) + affinity_inter.at<float>(idx_b, idx_a));
		//		//affinity_intra[idx_b] = 0;

		//		/// update inter affinity
		//		// update A_merged->c
		//		affinity_inter.col(idx_a) = affinity_inter.col(idx_a) + affinity_inter.col(idx_b);
		//		// update A_c->merged
		//		/* fast algorithm */
		//		// affinity_inter.row(idx_a) = rat * affinity_inter.row(idx_a) + (1 - rat) * affinity_inter.row(idx_b);


		//		for (int64 i = 0; i < label_indice.size(); ++i) {
		//			if (i == idx_a || label_indice[i].size() == 0) {
		//				continue;
		//			}

		//			float affinity_inter_i_merged = 0;

		//			if ((label_indice[i].size() + label_indice[idx_a].size()) > 500) {
		//				// if (0) {
		//				// convert to product of four matrices
		//				int64 msize = label_indice[i].size() * label_indice[idx_a].size();
		//				float* W_samples_sub_idx_a_i = new float[msize];
		//				float* W_samples_sub_i_idx_a = new float[msize];
		//				for (int m = 0; m < label_indice[idx_a].size(); ++m) {
		//					for (int n = 0; n < label_indice[i].size(); ++n) {
		//						W_samples_sub_idx_a_i[m * label_indice[i].size() + n] = W_samples_vec[int64(label_indice[idx_a][m]) * int64(W_samples.cols)
		//							+ int64(label_indice[i][n])];
		//						W_samples_sub_i_idx_a[n * label_indice[idx_a].size() + m] = W_samples_vec[int64(label_indice[i][n]) * int64(W_samples.cols)
		//							+ int64(label_indice[idx_a][m])];
		//					}
		//				}
		//				affinity_inter_i_merged = computeAc2merged(W_samples_sub_idx_a_i, W_samples_sub_i_idx_a,
		//					label_indice[idx_a].size(), label_indice[i].size());
		//				delete[]W_samples_sub_idx_a_i;
		//				delete[]W_samples_sub_i_idx_a;
		//			}
		//			else {
		//				for (int m = 0; m < label_indice[i].size(); ++m) {

		//					float s_W_c_idx_a_i = 0;
		//					for (int n = 0; n < label_indice[idx_a].size(); ++n) {
		//						// s_W_c_idx_a_i += W_samples.at<float>(label_indice[idx_a][n], label_indice[i][m]);
		//						s_W_c_idx_a_i += W_samples_vec[int64(label_indice[idx_a][n]) * int64(W_samples.cols) + int64(label_indice[i][m])];
		//					}

		//					float s_W_c_i_idx_a = 0;
		//					for (int n = 0; n < label_indice[idx_a].size(); ++n) {
		//						// s_W_c_i_idx_a += W_samples.at<float>(label_indice[i][m], label_indice[idx_a][n]);
		//						s_W_c_i_idx_a += W_samples_vec[int64(label_indice[i][m]) * int64(W_samples.cols) + int64(label_indice[idx_a][n])];
		//					}

		//					affinity_inter_i_merged += s_W_c_idx_a_i * s_W_c_i_idx_a;
		//				}
		//			}
		//			affinity_inter.at<float>(idx_a, i) = affinity_inter_i_merged;
		//			// if (A_a_b.at<float>(i, idx_a) != A_c_i_idx_a)
		//			//	cout << "wrong calculation" << endl;
		//			// A_a_b.at<float>(idx_a, i) = A_c_i_idx_a;
		//		}


		//		// affinity_inter.row(idx_a) = affinity_inter.row(idx_a) + affinity_inter.row(idx_b);
		//		//for (int m = 0; m < s.cols; ++m) {
		//		//	if (s.at<float>(m) != 0 && affinity_inter.at<float>(idx_a, m) != 0) {
		//		//		LOG(INFO) << s.at<float>(m) << ", " << affinity_inter.at<float>(idx_a, m);
		//		//	}
		//		//}
		//		// update A_c->idx_b
		//		affinity_inter.row(idx_b) = 0;

		//		// update A_idx_b->c
		//		affinity_inter.col(idx_b) = 0;

		//		// update A_merged<->c
		//		// cv::transpose(affinity_inter, affinity_inter_T);
		//		// affinity_inter_sym.row(idx_a) = affinity_inter.row(idx_a) + affinity_inter.col(idx_a).t();
		//		// affinity_inter_sym.col(idx_a) = affinity_inter.col(idx_a) + affinity_inter.row(idx_a).t();

		//		for (int64 i = 0; i < affinity_inter_sym.cols; ++i) {
		//			if (i == idx_a || label_indice[idx_a].size() == 0 || label_indice[i].size() == 0) {
		//				affinity_inter_sym.at<float>(idx_a, i) = 0;
		//				affinity_inter_sym_vec[int64(idx_a) * int64(affinity_inter_sym.cols) + i] = 0;
		//			}
		//			else if (i < idx_a) {
		//				affinity_inter_sym.at<float>(i, idx_a) =
		//					affinity_inter.at<float>(idx_a, i) * decm.at<float>(idx_a)
		//					+affinity_inter.at<float>(i, idx_a) * decm.at<float>(i);
		//				affinity_inter_sym_vec[int64(i) * int64(affinity_inter_sym.cols) + idx_a] = affinity_inter_sym.at<float>(i, idx_a);
		//			}
		//			else if (i > idx_a) {
		//				affinity_inter_sym.at<float>(idx_a, i) =
		//					affinity_inter.at<float>(idx_a, i) * decm.at<float>(idx_a)
		//					+affinity_inter.at<float>(i, idx_a) * decm.at<float>(i);
		//				affinity_inter_sym_vec[int64(idx_a) * int64(affinity_inter_sym.cols) + i] = affinity_inter_sym.at<float>(idx_a, i);
		//			}
		//		}

		//		for (int64 i = 0; i < affinity_inter_sym.cols; ++i) {
		//			affinity_inter_sym_vec[int64(idx_b) * int64(affinity_inter_sym.cols) + i] = 0;
		//			affinity_inter_sym_vec[int64(i) * int64(affinity_inter_sym.cols) + idx_b] = 0;
		//		}
		//		//for (int i = 0; i < affinity_inter_sym.rows; ++i) {
		//		//	if (idx_a == i || label_indice[i].size() == 0 || label_indice[idx_a].size() == 0) {
		//		//		affinity_inter_sym.at<float>(i, idx_a) = 0;
		//		//	}
		//		//	else {
		//		//		affinity_inter_sym.at<float>(i, idx_a) =
		//		//			affinity_inter.at<float>(i, j) / label_indice[i].size() / label_indice[i].size()
		//		//			+ affinity_inter_T.at<float>(i, j) / label_indice[j].size() / label_indice[j].size();
		//		//	}
		//		//}

		//		// update A_c->idx_b
		//		affinity_inter_sym.row(idx_b) = 0;
		//		// update A_idx_b->c
		//		affinity_inter_sym.col(idx_b) = 0;

		//		// update loss 

		//		// measure the NMI
		//		--num_categories;
		//	}
		//}

		while (1) {
			/* find the optimal cluster pair for merging    */
			/* previously, i traversed all clusters t0 seek */
			/* the best choice, actually, because each time */
			/* only one cluter is changed, we only need to  */
			/* update it loss an then compare with all other*/

			//for (int i = 0; i < affinity_inter_sym.rows; ++i) {
			//	cv::minMaxIdx(affinity_inter_sym.row(i), &minVal, &maxVal, minLoc, maxLoc);
			//	// get the indice of classes to merge
			//	int idx_a = maxLoc[0];
			//	int idx_b = maxLoc[1];
			//	maxLocs[i] = idx_b;
			//	float delta(0.0);
			//	for (int j = 0; j < affinity_inter_sym.cols - 1; ++j) {
			//		delta += MAX(0, 0.7 * maxVal - affinity_inter_sym.at<float>(0, j + 1));
			//	}
			//	if (is_final_eval_) {
			//		deltas[i] = maxVal;
			//	}
			//	else {
			//		deltas[i] = delta / num_categories + maxVal;
			//	}
			//}
			int idx_a, idx_b;
			// if (affinity_inter_sym.rows > 2000) {
			if (0) {
				int K_c = (num_categories - 1) / 100;

				//for (int i = 0; i < affinity_inter_sym.rows; ++i) {
				//	for (int j = 0; j < affinity_inter_sym.cols; ++j) {
				//		affinity_inter_sym_vec[i * affinity_inter_sym.rows + j] = affinity_inter_sym.at<float>(i, j);
				//	}
				//}
				std::pair<int, int> cluster_pair = seekOptimalPairs(affinity_inter_sym_vec, affinity_inter_sym.rows,
					affinity_inter_sym.cols, K_c + 1, 1.0);

				idx_a = cluster_pair.first;
				idx_b = cluster_pair.second;
			}
			else {
				// cv::Mat affinity_inter_sym_T, affinity_inter_sym_plus;
				// cv::transpose(affinity_inter_sym, affinity_inter_sym_T);
				// affinity_inter_sym_plus = affinity_inter_sym + affinity_inter_sym_T;
				for (int64 i = 0; i < affinity_inter_sym.rows; ++i) {
					cv::minMaxIdx(affinity_inter_sym.row(i), &minVal, &maxVal, minLoc, maxLoc);
					// get the indice of classes to merge
					int idx_a = maxLoc[0];
					int idx_b = maxLoc[1];
					maxLocs[i] = idx_b;

					cv::Mat affinity_inter_sym_row_i_sorted;
					// cv::sort(affinity_inter_sym_plus.row(i), affinity_inter_sym_row_i_sorted, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
					cv::sort(affinity_inter_sym.row(i), affinity_inter_sym_row_i_sorted, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

					float delta(0.0);
					//for (int j = 0; j < affinity_inter_sym.rows; ++j) {
					//	if (j != idx_b) {
					//		delta += maxVal - affinity_inter_sym.at<float>(i, j);
					//			// MAX(0.5 - (maxVal - affinity_inter_sym.at<float>(i, j)), 0); // the larger this value, the better
					//	}
					//}
					int knn = min((num_categories - 1), K_c_);
					int num_nonzero = 0;
					for (int j = 0; j < knn; ++j) {
						// if (affinity_inter_sym_row_i_sorted.at<float>(0, j + 1) != 0) {
							delta += (maxVal - affinity_inter_sym_row_i_sorted.at<float>(0, j + 1));
							// ++num_nonzero;
						// }
						// MAX(0.5 - (maxVal - affinity_inter_sym.at<float>(i, j)), 0); // the larger this value, the better					
					}
					// if (num_nonzero != 0) {
					delta /= knn;
					// }
					if (with_local_contrast_) {
						deltas[i] = maxVal + lcc_weight_ * delta;
						// LOG(INFO) << maxVal << " " << delta;
					}
					else {
						deltas[i] = maxVal + 0.0 * delta;
					}
				}
				// find the largest delta among deltas
				int pos = max_element(deltas.begin(), deltas.end()) - deltas.begin();
				float delta_max = deltas[pos];
				if (delta_max == 0) { // this means that no need to further merge any two clusters
					// propagate label_indice to opt_labels
					for (int i = 0; i < label_indice.size(); ++i) {
						for (int k = 0; k < label_indice[i].size(); ++k) {
							updated_labels[label_indice[i][k]] = i;
						}
					}
					float nmi = measure_nmi_ ? measure_.m_nmi_fast(label_gt_, updated_labels, num_samples) : 0;
					LOG(INFO) << "NCat " << num_categories << ": NMI = " << nmi << ", " << idx_a << " + " << idx_b;

					// in this case, we need to increase the K_s
					K_s_ = 2 * K_s_;
					break;
				}
				// merge two clusters
				idx_a = min(pos, maxLocs[pos]);
				idx_b = max(pos, maxLocs[pos]);
			}

			// update label_indice, merge idx_b to idx_a, and then clear idx_b
			float rat = float(label_indice[idx_a].size()) / float(label_indice[idx_a].size() + label_indice[idx_b].size());
			label_indice[idx_a].insert(label_indice[idx_a].end(), label_indice[idx_b].begin(), label_indice[idx_b].end());
			label_indice[idx_b].clear();
			decm.at<float>(idx_b) = 0;
			decm.at<float>(idx_a) = 1.0f / label_indice[idx_a].size() / label_indice[idx_a].size();

			//// update the intra affinity and inter affinity

			/// update intra affinity
			//affinity_intra[idx_a] = (affinity_intra[idx_a] + affinity_intra[idx_b] +
			//	affinity_inter.at<float>(idx_a, idx_b) + affinity_inter.at<float>(idx_b, idx_a));
			//affinity_intra[idx_b] = 0;

			/// update inter affinity
			// update A_merged->c
			affinity_inter.col(idx_a) = affinity_inter.col(idx_a) + affinity_inter.col(idx_b);
			// update A_c->merged
			/* fast algorithm */
			// affinity_inter.row(idx_a) = affinity_inter.row(idx_a) + affinity_inter.row(idx_b);

			
			for (int64 i = 0; i < label_indice.size(); ++i) {
				if (i == idx_a || label_indice[i].size() == 0) {
					continue;
				}

				float affinity_inter_i_merged = 0;

				if ((label_indice[i].size() + label_indice[idx_a].size()) > 500) {
				// if (0) {
					// convert to product of four matrices
					int64 msize = label_indice[i].size() * label_indice[idx_a].size();
					float* W_samples_sub_idx_a_i = new float[msize];
					float* W_samples_sub_i_idx_a = new float[msize];
					for (int m = 0; m < label_indice[idx_a].size(); ++m) {
						for (int n = 0; n < label_indice[i].size(); ++n) {
							W_samples_sub_idx_a_i[m * label_indice[i].size() + n] = W_samples_vec[int64(label_indice[idx_a][m]) * int64(W_samples.cols)
								+ int64(label_indice[i][n])];
							W_samples_sub_i_idx_a[n * label_indice[idx_a].size() + m] = W_samples_vec[int64(label_indice[i][n]) * int64(W_samples.cols)
								+ int64(label_indice[idx_a][m])];
						}
					}
					affinity_inter_i_merged = computeAc2merged(W_samples_sub_idx_a_i, W_samples_sub_i_idx_a,
						label_indice[idx_a].size(), label_indice[i].size());
					delete[]W_samples_sub_idx_a_i;
					delete[]W_samples_sub_i_idx_a;
				}
				else {
					for (int m = 0; m < label_indice[i].size(); ++m) {

						float s_W_c_idx_a_i = 0;
						for (int n = 0; n < label_indice[idx_a].size(); ++n) {
							// s_W_c_idx_a_i += W_samples.at<float>(label_indice[idx_a][n], label_indice[i][m]);
							s_W_c_idx_a_i += W_samples_vec[int64(label_indice[idx_a][n]) * int64(W_samples.cols) + int64(label_indice[i][m])];
						}

						float s_W_c_i_idx_a = 0;
						for (int n = 0; n < label_indice[idx_a].size(); ++n) {
							// s_W_c_i_idx_a += W_samples.at<float>(label_indice[i][m], label_indice[idx_a][n]);
							s_W_c_i_idx_a += W_samples_vec[int64(label_indice[i][m]) * int64(W_samples.cols) + int64(label_indice[idx_a][n])];
						}

						affinity_inter_i_merged += s_W_c_idx_a_i * s_W_c_i_idx_a;
					}
				}
				affinity_inter.at<float>(idx_a, i) = affinity_inter_i_merged;
				// if (A_a_b.at<float>(i, idx_a) != A_c_i_idx_a)
				//	cout << "wrong calculation" << endl;
				// A_a_b.at<float>(idx_a, i) = A_c_i_idx_a;
			}
			

			// affinity_inter.row(idx_a) = affinity_inter.row(idx_a) + affinity_inter.row(idx_b);
			//for (int m = 0; m < s.cols; ++m) {
			//	if (s.at<float>(m) != 0 && affinity_inter.at<float>(idx_a, m) != 0) {
			//		LOG(INFO) << s.at<float>(m) << ", " << affinity_inter.at<float>(idx_a, m);
			//	}
			//}
			// update A_c->idx_b
			affinity_inter.row(idx_b) = 0;

			// update A_idx_b->c
			affinity_inter.col(idx_b) = 0;

			// update A_merged<->c
			// cv::transpose(affinity_inter, affinity_inter_T);
			// affinity_inter_sym.row(idx_a) = affinity_inter.row(idx_a) + affinity_inter.col(idx_a).t();
			// affinity_inter_sym.col(idx_a) = affinity_inter.col(idx_a) + affinity_inter.row(idx_a).t();

			for (int64 i = 0; i < affinity_inter_sym.cols; ++i) {
				if (i == idx_a || label_indice[idx_a].size() == 0 || label_indice[i].size() == 0) {
					affinity_inter_sym.at<float>(idx_a, i) = 0;
					affinity_inter_sym_vec[int64(idx_a) * int64(affinity_inter_sym.cols) + i] = 0;
				}
				else if (i < idx_a) {
					affinity_inter_sym.at<float>(i, idx_a) =
						affinity_inter.at<float>(idx_a, i) * decm.at<float>(idx_a)
						+ affinity_inter.at<float>(i, idx_a) * decm.at<float>(i);
					affinity_inter_sym_vec[int64(i) * int64(affinity_inter_sym.cols) + idx_a] = affinity_inter_sym.at<float>(i, idx_a);
				}
				else if (i > idx_a) {
					affinity_inter_sym.at<float>(idx_a, i) =
						affinity_inter.at<float>(idx_a, i) * decm.at<float>(idx_a)
						+ affinity_inter.at<float>(i, idx_a) * decm.at<float>(i);
					affinity_inter_sym_vec[int64(idx_a) * int64(affinity_inter_sym.cols) + i] = affinity_inter_sym.at<float>(idx_a, i);
				}
			}

			for (int64 i = 0; i < affinity_inter_sym.cols; ++i) {
				affinity_inter_sym_vec[int64(idx_b) * int64(affinity_inter_sym.cols) + i] = 0;
				affinity_inter_sym_vec[int64(i) * int64(affinity_inter_sym.cols) + idx_b] = 0;
			}
			//for (int i = 0; i < affinity_inter_sym.rows; ++i) {
			//	if (idx_a == i || label_indice[i].size() == 0 || label_indice[idx_a].size() == 0) {
			//		affinity_inter_sym.at<float>(i, idx_a) = 0;
			//	}
			//	else {
			//		affinity_inter_sym.at<float>(i, idx_a) =
			//			affinity_inter.at<float>(i, j) / label_indice[i].size() / label_indice[i].size()
			//			+ affinity_inter_T.at<float>(i, j) / label_indice[j].size() / label_indice[j].size();
			//	}
			//}

			// update A_c->idx_b
			affinity_inter_sym.row(idx_b) = 0;
			// update A_idx_b->c
			affinity_inter_sym.col(idx_b) = 0;

			// update loss 

			// measure the NMI
			--num_categories;
			if (num_categories <= num_final_class_) {
				// propagate label_indice to opt_labels
				for (int i = 0; i < label_indice.size(); ++i) {
					for (int k = 0; k < label_indice[i].size(); ++k) {
						updated_labels[label_indice[i][k]] = i;
					}
				}
				float nmi = measure_nmi_ ? measure_.m_nmi_fast(label_gt_, updated_labels, num_samples) : 0;
				LOG(INFO) << "NCat " << num_categories << ": NMI = " << nmi << ", " << idx_a << " + " << idx_b;
				break;
			}
			if (iter % test_iter == 100) {
				// propagate label_indice to opt_labels
				for (int i = 0; i < label_indice.size(); ++i) {
					for (int k = 0; k < label_indice[i].size(); ++k) {
						updated_labels[label_indice[i][k]] = i;
					}
				}
				float nmi = measure_nmi_ ? measure_.m_nmi_fast(label_gt_, updated_labels, num_samples) : 0;
				LOG(INFO) << "NCat " << num_categories << ": NMI = " << nmi << ", " << idx_a << " + " << idx_b;
			}
			if (/*delta_loss[pos] < 0 || */num_categories <= floor(ratio_stage_ * label_indice.size()))
				break;
			++iter;
		}

		delete[]W_samples_vec;
		delete[]affinity_inter_sym_vec;
		delete[]updated_labels;
	}

	// added by Jianwei Yang @ 09/28/2015
	template <typename Dtype>
	void Solver<Dtype>::OptimizeLabels(vector<vector<int64> >& label_indice, cv::Mat& W_samples, int topK) {
		MergeClusters(label_indice, W_samples);
		return;
		int num_samples = W_samples.rows;
		// compute intra_affinity, A(C) = \sum_m \sum_n W_m_n * W_n_m, m, n \in C
		vector<float> affinity_intra(label_indice.size());
		for (int i = 0; i < label_indice.size(); ++i) {
			float val = 0;
			for (int m = 0; m < label_indice[i].size(); ++m) {
				for (int n = 0; n < label_indice[i].size(); ++n) {
					if (m == n)
						continue;
					float entry_m_n = W_samples.at<float>(label_indice[i][m], label_indice[i][n]);
					float entry_n_m = W_samples.at<float>(label_indice[i][n], label_indice[i][m]);
					val += entry_m_n * entry_n_m;
				}
			}

			affinity_intra[i] = val;
		}

		// compute inter_affinity = A(C_i -> C_j)
		cv::Mat affinity_inter = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
		for (int i = 0; i < label_indice.size(); ++i) {
			for (int j = 0; j < label_indice.size(); ++j) {
				if (i == j) {
					affinity_inter.at<float>(j, i) = 0;
					continue;
				}
				float A_c_i_j = 0;

				for (int m = 0; m < label_indice[i].size(); ++m) {

					float s_W_c_j_i = 0;
					for (int n = 0; n < label_indice[j].size(); ++n) {
						s_W_c_j_i += W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
					}

					float s_W_c_i_j = 0;
					for (int n = 0; n < label_indice[j].size(); ++n) {
						s_W_c_i_j += W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
					}

					A_c_i_j += s_W_c_j_i * s_W_c_i_j;
				}

				//for (int m = 0; m < label_indice[i].size(); ++m) {
				//	for (int n = 0; n < label_indice[j].size(); ++n) {
				//		float entry_m_n = W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
				//		float entry_n_m = W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
				//		A_c_i_j += entry_m_n * entry_n_m;
				//	}
				//}

				affinity_inter.at<float>(j, i) = A_c_i_j;
			}
		}

		// compute recent loss, Loss = L_intra + L_inter
		// L_intra = 1 / |C| * \sum_c (1 - mean(A(c)))
		// L_extra = 1 / |C|^2 * \sum_i \sum_j A(C_i, C_j)
		float loss(0), loss_intra(0), loss_inter(0);
		// loss += loss_intra
		for (int i = 0; i < affinity_intra.size(); ++i) {
			loss_intra += 1 - affinity_intra[i] / label_indice[i].size() / (label_indice[i].size() - 1);
		}

		// loss += loss_extra
		for (int i = 0; i < label_indice.size(); ++i) {
			for (int j = 0; j < label_indice.size(); ++j) {
				loss_inter += affinity_inter.at<float>(j, i) / label_indice[j].size() / label_indice[j].size();  // A_c_i_j
			}
		}

		loss = loss_intra + loss_inter;

		// find optimal cluster pair so that the loss decrease mostly.
		// we are not going to traverse all possible pairs because it
		// is exhausted if there are thousands of clusters, alternatively,
		// we find top-K candidates with largest inter_affinity
		cv::Mat affinity_inter_sym = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
		cv::Mat affinity_inter_T;
		cv::transpose(affinity_inter, affinity_inter_T);

		for (int i = 0; i < affinity_inter_sym.rows; ++i) {
			for (int j = 0; j < affinity_inter_sym.cols; ++j) {
				if (j <= i) {
					affinity_inter_sym.at<float>(i, j) = 0;
				}
				else {
					affinity_inter_sym.at<float>(i, j) = affinity_inter.at<float>(i, j) / label_indice[i].size() / label_indice[i].size() 
						+ affinity_inter_T.at<float>(i, j) / label_indice[j].size() / label_indice[j].size();;
				}
			}
		}
		int test_iter = 100;
		int iter = 0;
		int num_categories = label_indice.size();
		double minVal, maxVal;
		int minLoc[2], maxLoc[2];
		int* updated_labels = new int[num_samples];

		float loss_intra_new, loss_inter_new;
		cv::Mat affinity_inter_sym_temp;
		while (1) { // we will continously find optimal pair until the loss do not decrease as much as a pre-defined extent
			// find highest K entries in A
			vector<int> locations(topK, 0);
			int k = 0;
			vector<float> delta_loss(topK);
			vector<pair<int, int> > pairs_topK(0);
			vector<pair<float, float> > losses_new_topK(0);
			affinity_inter_sym.copyTo(affinity_inter_sym_temp);
			while (k < topK) {
				cv::minMaxIdx(affinity_inter_sym_temp, &minVal, &maxVal, minLoc, maxLoc);
				// get the indice of classes to merge
				int idx_a = maxLoc[0];
				int idx_b = maxLoc[1];
				// set the value at [idx_a, idx_b] be zero
				affinity_inter_sym_temp.at<float>(idx_a, idx_b) = 0;
				pairs_topK.push_back(make_pair(idx_a, idx_b));

				//// compute new loss after merging idx_a and idx_b clusters

				/// compute new intra loss

				// remove loss from original two clusters
				loss_intra_new = loss_intra;
				loss_intra_new -= (1 - affinity_intra[idx_a] / label_indice[idx_a].size() / (label_indice[idx_a].size() - 1));
				loss_intra_new -= (1 - affinity_intra[idx_b] / label_indice[idx_b].size() / (label_indice[idx_b].size() - 1));

				// add loss when merging them 
				float affinity_combine = (affinity_intra[idx_a] + affinity_intra[idx_b] + 
					affinity_inter.at<float>(idx_a, idx_b) + affinity_inter.at<float>(idx_b, idx_a));
				int size_combine = label_indice[idx_a].size() + label_indice[idx_b].size();
				loss_intra_new += (1 - affinity_combine / size_combine / (size_combine - 1));

				/// compute new inter loss

				// remove inter loss from original two clusters
				loss_inter_new = loss_inter;
				for (int i = 0; i < label_indice.size(); ++i) {
					loss_inter_new -= affinity_inter.at<float>(idx_a, i) / label_indice[idx_a].size() / label_indice[idx_a].size();
					loss_inter_new -= affinity_inter.at<float>(idx_b, i) / label_indice[idx_b].size() / label_indice[idx_b].size();
				}

				// add inter loss when combining two clusters
				// recompute the inter_loss from C(idx_a, idx_b)
				vector<int64> label_indice_merged = label_indice[idx_a];
				label_indice_merged.insert(label_indice_merged.end(), label_indice[idx_b].begin(), label_indice[idx_b].end());
				for (int i = 0; i < label_indice.size(); ++i) {
					if (i == idx_a) {

						continue;
					}

					float affinity_inter_i_merged = 0;
					for (int m = 0; m < label_indice[i].size(); ++m) {

						float s_W_c_idx_a_i = 0;
						for (int n = 0; n < label_indice_merged.size(); ++n) {
							s_W_c_idx_a_i += W_samples.at<float>(label_indice_merged[n], label_indice[i][m]);
						}

						float s_W_c_i_idx_a = 0;
						for (int n = 0; n < label_indice_merged.size(); ++n) {
							s_W_c_i_idx_a += W_samples.at<float>(label_indice[i][m], label_indice_merged[n]);
						}

						affinity_inter_i_merged += s_W_c_idx_a_i * s_W_c_i_idx_a;
					}
					//for (int m = 0; m < label_indice[i].size(); ++m) {
					//	for (int n = 0; n < label_indice_merged.size(); ++n) {
					//		float entry_m_n = W_samples.at<float>(label_indice[i][m], label_indice_merged[n]);
					//		float entry_n_m = W_samples.at<float>(label_indice_merged[n], label_indice[i][m]);
					//		affinity_inter_i_merged += entry_m_n * entry_n_m;
					//	}
					//}

					loss_inter_new += affinity_inter_i_merged / label_indice_merged.size() / label_indice_merged.size();
					// if (A_a_b.at<float>(i, idx_a) != A_c_i_idx_a)
					//	cout << "wrong calculation" << endl;
					// A_a_b.at<float>(idx_a, i) = A_c_i_idx_a;
				}
				delta_loss[k] = (loss_intra / (num_categories) + loss_inter / num_categories / (num_categories - 1)) 
					  - (loss_intra_new / (num_categories - 1) + loss_inter_new / (num_categories - 1) / (num_categories - 2));
				// delta_loss[k] = (loss_intra + loss_inter)
					  // - (loss_intra_new + loss_inter_new);
				// delta_loss[k] = loss_intra / num_categories - loss_intra_new / (num_categories - 1);
				losses_new_topK.push_back(make_pair(loss_intra_new, loss_inter_new));
				++k;
			}

			// find the pair with most loss decrease
			int pos = max_element(delta_loss.begin(), delta_loss.end()) - delta_loss.begin();
			if (/*delta_loss[pos] < 0 || */num_categories < 0.1 * label_indice.size())
				break;

			int idx_a = pairs_topK[pos].first;
			int idx_b = pairs_topK[pos].second;

			// update label_indice, merge idx_b to idx_a, and then clear idx_b
			label_indice[idx_a].insert(label_indice[idx_a].end(), label_indice[idx_b].begin(), label_indice[idx_b].end());
			label_indice[idx_b].clear();

			//// update the intra affinity and inter affinity

			/// update intra affinity
			affinity_intra[idx_a] = (affinity_intra[idx_a] + affinity_intra[idx_b] +
				affinity_inter.at<float>(idx_a, idx_b) + affinity_inter.at<float>(idx_b, idx_a));
			affinity_intra[idx_b] = 0;

			/// update inter affinity
			// update A_merged->c
			affinity_inter.col(idx_a) = affinity_inter.col(idx_a) + affinity_inter.col(idx_b);
			// update A_c->merged
			for (int i = 0; i < label_indice.size(); ++i) {
				if (i == idx_a) {

					continue;
				}

				float affinity_inter_i_merged = 0;

				for (int m = 0; m < label_indice[i].size(); ++m) {

					float s_W_c_idx_a_i = 0;
					for (int n = 0; n < label_indice[idx_a].size(); ++n) {
						s_W_c_idx_a_i += W_samples.at<float>(label_indice[idx_a][n], label_indice[i][m]);
					}

					float s_W_c_i_idx_a = 0;
					for (int n = 0; n < label_indice[idx_a].size(); ++n) {
						s_W_c_i_idx_a += W_samples.at<float>(label_indice[i][m], label_indice[idx_a][n]);
					}

					affinity_inter_i_merged += s_W_c_idx_a_i * s_W_c_i_idx_a;
				}

				//for (int m = 0; m < label_indice[i].size(); ++m) {
				//	for (int n = 0; n < label_indice[idx_a].size(); ++n) {
				//		float entry_m_n = W_samples.at<float>(label_indice[i][m], label_indice[idx_a][n]);
				//		float entry_n_m = W_samples.at<float>(label_indice[idx_a][n], label_indice[i][m]);
				//		affinity_inter_i_merged += entry_m_n * entry_n_m;
				//	}
				//}

				affinity_inter.at<float>(idx_a, i) = affinity_inter_i_merged;
				// if (A_a_b.at<float>(i, idx_a) != A_c_i_idx_a)
				//	cout << "wrong calculation" << endl;
				// A_a_b.at<float>(idx_a, i) = A_c_i_idx_a;
			}
			// update A_c->idx_b
			affinity_inter.row(idx_b) = 0;

			// update A_idx_b->c
			affinity_inter.col(idx_b) = 0;

			// update A_merged<->c
			cv::transpose(affinity_inter, affinity_inter_T);
			for (int i = 0; i < affinity_inter_sym.rows; ++i) {
				for (int j = 0; j < affinity_inter_sym.cols; ++j) {
					if (j <= i || label_indice[i].size() == 0 || label_indice[j].size() == 0) {
						affinity_inter_sym.at<float>(i, j) = 0;
					}
					else {
						affinity_inter_sym.at<float>(i, j) = 
							affinity_inter.at<float>(i, j) / label_indice[i].size() / label_indice[i].size() 
							+ affinity_inter_T.at<float>(i, j) / label_indice[j].size() / label_indice[j].size();
					}
				}
			}
			// update A_c->idx_b
			affinity_inter_sym.row(idx_b) = 0;
			// update A_idx_b->c
			affinity_inter_sym.col(idx_b) = 0;

			// update loss 
			loss_intra = losses_new_topK[pos].first;
			loss_inter = losses_new_topK[pos].second;
			loss = loss_intra + loss_inter;

			// measure the NMI
			// propagate label_indice to opt_labels
			for (int i = 0; i < label_indice.size(); ++i) {
				for (int k = 0; k < label_indice[i].size(); ++k) {
					updated_labels[label_indice[i][k]] = i;
				}
			}
			--num_categories;
			if (num_categories == num_final_class_) {
				float nmi = measure_.m_nmi_fast(label_gt_, updated_labels, num_samples);
				LOG(INFO) << "NCat " << num_categories << ": NMI = " << nmi << ", dLoss = " << delta_loss[pos] << ", " << idx_a << " + " << idx_b;
				break;
			}
			if (iter % test_iter == 0) {
				float nmi = measure_.m_nmi_fast(label_gt_, updated_labels, num_samples);
				LOG(INFO) << "NCat " << num_categories << ": NMI = " << nmi << ", dLoss = " << delta_loss[pos] << ", " << idx_a << " + " << idx_b;
			}
			++iter;
		}
	}

	// added by Jianwei Yang @10/29/2015
	// main body for finding optimal labels for training
	/*========================================*/
	/* y* = argmin_{y} L(F_{t-1}, y_{t-1}, y) */
	/*========================================*/
	template <typename Dtype>
	void Solver<Dtype>::FindOptimalLables(cv::Mat& feat_cvmat, float* feat_ptr, int K, int a, int topK, int* opt_labels) {
		int num_samples = feat_cvmat.rows;
		cv::Mat D_samples, nIdx;
		float* D_samples_ptr = new float[int64(feat_cvmat.rows) * int64((K + 1))];
		int* nIdx_ptr = new int[int64(feat_cvmat.rows) * int64((K + 1))];
		batchDistance(feat_ptr, D_samples_ptr, nIdx_ptr, (K + 1), feat_cvmat.rows, feat_cvmat.cols);
		nIdx = cv::Mat(feat_cvmat.rows, (K + 1), CV_32SC1, nIdx_ptr, 0);
		D_samples = cv::Mat(feat_cvmat.rows, (K + 1), CV_32FC1, D_samples_ptr, 0);

		// compute distance between samples
		//cv::Mat D_samples, nIdx;		
		//cv::batchDistance(feat_cvmat, feat_cvmat, D_samples, CV_32FC1, nIdx, cv::NORM_L2, K + 1);

		// compute sigma based on K-NNs
		float sigma_square = 0;
		for (int i = 0; i < feat_cvmat.rows; ++i) {
			D_samples.at<float>(i, 0) = 0;
			for (int k = 0; k < K; ++k) {
				// float factor = cv::norm(feat_cvmat.row(i) - feat_cvmat_mean, cv::NORM_L2);  // measure the bias to center
				// D_samples.at<float>(i, k + 1) = cv::norm(feat_cvmat.row(i) - feat_cvmat.row(nIdx.at<int>(i, k + 1)));
				sigma_square += D_samples.at<float>(i, k + 1) /** D_samples.at<float>(i, k + 1)*/;
			}
		}
		sigma_square /= feat_cvmat.rows * K;
		sigma_square *= a;
		LOG(INFO) << "sigma_sq: " << sigma_square;
		// compute affinity matrix
		cv::Mat W_samples(feat_cvmat.rows, feat_cvmat.rows, CV_32FC1, cvScalarAll(0.0));
		for (int i = 0; i < feat_cvmat.rows; ++i) {
			for (int k = 0; k < K + 1; ++k) {
				if (i == nIdx.at<int>(i, k))
					continue;
				W_samples.at<float>(i, nIdx.at<int>(i, k)) = exp(-D_samples.at<float>(i, k) /** D_samples.at<float>(i, k)*/ / sigma_square);
			}
		}

		// float* D_samples_ptr = new float[int64(feat_cvmat.rows) * int64(feat_cvmat.rows)];

		// entityDistance(feat_ptr, D_samples_ptr, feat_cvmat.rows, feat_cvmat.cols);

		if (epoch_ < skip_order_) { // if it is the first epoch, then we use cluster algorithms for initialization, e.g., k-means, for the sake of simplification
			if (path_label_ == "") {
				vector<int> visited(num_samples, -1);
				vector<float> affinities(num_samples, 0);
				vector<int> label_indice_p(0);
				int count = 0;
				for (int i = 0; i < nIdx.rows; ++i) {
					int cur_idx = i;
					label_indice_p.clear();
					while (visited[cur_idx] == -1) {
						label_indice_p.push_back(cur_idx);
						// if (label_indice_p.size() > 10)
							// break;
						int neighbor;
						for (int k = 0; k < nIdx.cols; ++k) {
							neighbor = nIdx.at<int>(cur_idx, k);
							if (nIdx.at<int>(cur_idx, k) != cur_idx)
								break;
						}
						visited[cur_idx] = -2;

						affinities[cur_idx] = W_samples.at<float>(cur_idx, neighbor);
						cur_idx = neighbor;
					}
					if (visited[cur_idx] < 0) {
						visited[cur_idx] = count;
						++count;
					}
					int cluster_id = visited[cur_idx];
					for (int k = 0; k < label_indice_p.size(); ++k) {
						visited[label_indice_p[k]] = cluster_id;
					}
				}

				// initialize labels, each sample has a different label at first
				vector<vector<int> > label_indice;
				label_indice.resize(count);
				for (int i = 0; i < num_samples; ++i) {
					label_indice[visited[i]].push_back(i);
				}
				int num_max = 0;
				for (int i = 0; i < label_indice.size(); ++i) {
					if (num_max < label_indice[i].size()) {
						num_max = label_indice[i].size();
					}
				}
				// K_NN_ = num_max;
				// propagate label_indice to opt_labels
				num_categ_ = 0;
				for (int i = 0; i < label_indice.size(); ++i) {
					if (label_indice[i].size() > 0) {
						++num_categ_;
					}
					for (int k = 0; k < label_indice[i].size(); ++k) {
						opt_labels[label_indice[i][k]] = i;
					}
				}
				if (from_beg_) {
					for (int i = 0; i < num_samples; ++i) {
						opt_labels[i] = i;
					}
				}
			}
			else {
				bmat readbmat;
				readbmat.read_bmat(path_label_, (char*)opt_labels);
				num_categ_ = 2572;
			}
			/*
			// k-means is too time-consuming
			cv::Mat labels, centers;
			cv::kmeans(feat_cvmat, num_samples / 3, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
			3, cv::KMEANS_PP_CENTERS, centers);
			memcpy(opt_labels, labels.data, num_samples * sizeof(int));
			*/
			/*
			if (_runOpts->initial_classes == num_samples) {
			label_indice.resize(num_samples);
			for (int i = 0; i < num_samples; ++i) {
			label_indice[i].push_back(i);
			}
			A = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
			A_a_b = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);

			cv::transpose(W_samples, W_samples_T);
			cv::multiply(W_samples_T, W_samples, A_a_b);
			}
			// else if (_runOpts->initial_classes > 0 && _runOpts->initial_classes < num_samples) {
			else if (_label_indice.size() == 0 || epoch <= 1) {
			// alternatively, we use p-link propagation as used in ECCV 2012
			vector<int> visited(num_samples, -1);
			vector<float> affinities(num_samples, 0);
			vector<int> label_indice_p(0);
			int count = 0;
			for (int i = 0; i < nIdx.rows; ++i) {
			int cur_idx = i;
			label_indice_p.clear();
			while (visited[cur_idx] == -1) {
			label_indice_p.push_back(cur_idx);
			int neighbor;
			for (int k = 0; k < nIdx.cols; ++k) {
			neighbor = nIdx.at<int>(cur_idx, k);
			if (nIdx.at<int>(cur_idx, k) != cur_idx)
			break;
			}
			visited[cur_idx] = -2;

			affinities[cur_idx] = W_samples.at<float>(cur_idx, neighbor);
			cur_idx = neighbor;
			}
			if (visited[cur_idx] < 0) {
			visited[cur_idx] = count;
			++count;
			}
			int cluster_id = visited[cur_idx];
			for (int k = 0; k < label_indice_p.size(); ++k) {
			visited[label_indice_p[k]] = cluster_id;
			}
			}

			// initialize labels, each sample has a different label at first
			label_indice.resize(count);
			for (int i = 0; i < num_samples; ++i) {
			label_indice[visited[i]].push_back(i);
			}
			// compute initial Affinity
			float affinity = 0;
			for (int i = 0; i < label_indice.size(); ++i) {
			float aff_p = 0;
			for (int k = 0; k < label_indice[i].size(); ++k) {
			aff_p += affinities[label_indice[i][k]];
			}
			affinity += aff_p; // / label_indice[i].size();
			}

			cout << "averrage initial affinity: " << affinity / num_samples << endl;

			A = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
			A_a_b = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
			for (int i = 0; i < label_indice.size(); ++i) {
			for (int j = 0; j < label_indice.size(); ++j) {
			// cv::Mat W_c_i_j(label_indice[i].size(), label_indice[j].size(), CV_32FC1);
			// cv::Mat W_c_j_i(label_indice[j].size(), label_indice[i].size(), CV_32FC1);

			float A_c_i_j = 0;
			//for (int m = 0; m < label_indice[i].size(); ++m) {
			//	for (int n = 0; n < label_indice[j].size(); ++n) {
			//		W_c_i_j.at<float>(m, n) = W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
			//		W_c_j_i.at<float>(n, m) = W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
			//	}
			//}

			for (int m = 0; m < label_indice[i].size(); ++m) {

			float s_W_c_j_i = 0;
			for (int n = 0; n < label_indice[j].size(); ++n) {
			s_W_c_j_i += W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
			}

			float s_W_c_i_j = 0;
			for (int n = 0; n < label_indice[j].size(); ++n) {
			s_W_c_i_j += W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
			}

			A_c_i_j += s_W_c_j_i * s_W_c_i_j;
			}

			// A_c_i_j = cv::sum(W_c_j_i * W_c_i_j).val[0];
			A_c_i_j /= label_indice[j].size() * label_indice[j].size();
			// if (A_a_b.at<float>(i, idx_a) != A_c_i_idx_a)
			//	cout << "wrong calculation" << endl;
			A_a_b.at<float>(j, i) = A_c_i_j;
			}
			}
			}
			*/
		}
		else { // if it is not the first epoch, we will find the optimal labels based on previous predictions
			// re-organize labels
			map<int, vector<int64> > label_indice_map;
			for (int i = 0; i < num_samples; ++i) {
				label_indice_map[opt_labels[i]].push_back(i);
			}
			vector<vector<int64> > label_indice;
			vector<int> labels;
			for (std::map<int, vector<int64>>::iterator it = label_indice_map.begin(); it != label_indice_map.end(); ++it) {
				labels.push_back(it->first);
				label_indice.push_back(it->second);
			}

			// optimize labels to reduce the loss
			OptimizeLabels(label_indice, W_samples, topK);
			num_categ_ = 0;
			// propagate label_indice to opt_labels
			for (int i = 0; i < label_indice.size(); ++i) {
				if (label_indice[i].size() > 0) {
					++num_categ_;
				}
				for (int k = 0; k < label_indice[i].size(); ++k) {
					opt_labels[label_indice[i][k]] = i; // labels[i];
				}
			}
		}
	}

	// added by Jianwei Yang @09/28/2015
	// main body for finding optimal labels for training
	/*========================================*/
	/* y* = argmin_{y} L(F_{t-1}, y_{t-1}, y) */
	/*========================================*/
	template <typename Dtype>
	void Solver<Dtype>::FindOptimalLables_noKnn(cv::Mat& feat_cvmat, float* feat_ptr, int K, int a, int topK, int* opt_labels) {
		int num_samples = feat_cvmat.rows;


		if (epoch_ < skip_order_) { // if it is the first epoch, then we use cluster algorithms for initialization, e.g., k-means, for the sake of simplification
			
			// k-means is too time-consuming
			//cv::Mat labels, centers;
			//cv::kmeans(feat_cvmat, num_samples / 10, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
			//	3, cv::KMEANS_PP_CENTERS, centers);
			//memcpy(opt_labels, labels.data, num_samples * sizeof(int));
			
			for (int i = 0; i < num_samples; ++i) {
				opt_labels[i] = i;
			}
			/*
			if (_runOpts->initial_classes == num_samples) {
			label_indice.resize(num_samples);
			for (int i = 0; i < num_samples; ++i) {
			label_indice[i].push_back(i);
			}
			A = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
			A_a_b = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);

			cv::transpose(W_samples, W_samples_T);
			cv::multiply(W_samples_T, W_samples, A_a_b);
			}
			// else if (_runOpts->initial_classes > 0 && _runOpts->initial_classes < num_samples) {
			else if (_label_indice.size() == 0 || epoch <= 1) {
			// alternatively, we use p-link propagation as used in ECCV 2012
			vector<int> visited(num_samples, -1);
			vector<float> affinities(num_samples, 0);
			vector<int> label_indice_p(0);
			int count = 0;
			for (int i = 0; i < nIdx.rows; ++i) {
			int cur_idx = i;
			label_indice_p.clear();
			while (visited[cur_idx] == -1) {
			label_indice_p.push_back(cur_idx);
			int neighbor;
			for (int k = 0; k < nIdx.cols; ++k) {
			neighbor = nIdx.at<int>(cur_idx, k);
			if (nIdx.at<int>(cur_idx, k) != cur_idx)
			break;
			}
			visited[cur_idx] = -2;

			affinities[cur_idx] = W_samples.at<float>(cur_idx, neighbor);
			cur_idx = neighbor;
			}
			if (visited[cur_idx] < 0) {
			visited[cur_idx] = count;
			++count;
			}
			int cluster_id = visited[cur_idx];
			for (int k = 0; k < label_indice_p.size(); ++k) {
			visited[label_indice_p[k]] = cluster_id;
			}
			}

			// initialize labels, each sample has a different label at first
			label_indice.resize(count);
			for (int i = 0; i < num_samples; ++i) {
			label_indice[visited[i]].push_back(i);
			}
			// compute initial Affinity
			float affinity = 0;
			for (int i = 0; i < label_indice.size(); ++i) {
			float aff_p = 0;
			for (int k = 0; k < label_indice[i].size(); ++k) {
			aff_p += affinities[label_indice[i][k]];
			}
			affinity += aff_p; // / label_indice[i].size();
			}

			cout << "averrage initial affinity: " << affinity / num_samples << endl;

			A = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
			A_a_b = cv::Mat(label_indice.size(), label_indice.size(), CV_32FC1);
			for (int i = 0; i < label_indice.size(); ++i) {
			for (int j = 0; j < label_indice.size(); ++j) {
			// cv::Mat W_c_i_j(label_indice[i].size(), label_indice[j].size(), CV_32FC1);
			// cv::Mat W_c_j_i(label_indice[j].size(), label_indice[i].size(), CV_32FC1);

			float A_c_i_j = 0;
			//for (int m = 0; m < label_indice[i].size(); ++m) {
			//	for (int n = 0; n < label_indice[j].size(); ++n) {
			//		W_c_i_j.at<float>(m, n) = W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
			//		W_c_j_i.at<float>(n, m) = W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
			//	}
			//}

			for (int m = 0; m < label_indice[i].size(); ++m) {

			float s_W_c_j_i = 0;
			for (int n = 0; n < label_indice[j].size(); ++n) {
			s_W_c_j_i += W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
			}

			float s_W_c_i_j = 0;
			for (int n = 0; n < label_indice[j].size(); ++n) {
			s_W_c_i_j += W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
			}

			A_c_i_j += s_W_c_j_i * s_W_c_i_j;
			}

			// A_c_i_j = cv::sum(W_c_j_i * W_c_i_j).val[0];
			A_c_i_j /= label_indice[j].size() * label_indice[j].size();
			// if (A_a_b.at<float>(i, idx_a) != A_c_i_idx_a)
			//	cout << "wrong calculation" << endl;
			A_a_b.at<float>(j, i) = A_c_i_j;
			}
			}
			}
			*/
		}
		else { // if it is not the first epoch, we will find the optimal labels based on previous predictions
			cv::Mat D_samples;
			float* D_samples_ptr = new float[int64(feat_cvmat.rows) * int64(feat_cvmat.rows)];
			entityDistance(feat_ptr, D_samples_ptr, feat_cvmat.rows, feat_cvmat.cols);
			D_samples = cv::Mat(feat_cvmat.rows, feat_cvmat.rows, CV_32FC1, D_samples_ptr, 0);

			// compute sigma based on K-NNs
			float sigma_square = 0;
			for (int i = 0; i < feat_cvmat.rows; ++i) {
				// D_samples.at<float>(i, 0) = 0;
				for (int k = 0; k < K; ++k) {
					// float factor = cv::norm(feat_cvmat.row(i) - feat_cvmat_mean, cv::NORM_L2);  // measure the bias to center
					// D_samples.at<float>(i, k + 1) = cv::norm(feat_cvmat.row(i) - feat_cvmat.row(nIdx.at<int>(i, k + 1)));
					sigma_square += D_samples.at<float>(i, k + 1) /** D_samples.at<float>(i, k + 1)*/;
				}
			}
			sigma_square /= feat_cvmat.rows * K;
			sigma_square *= a;
			LOG(INFO) << "sigma_sq: " << sigma_square;
			// compute affinity matrix
			cv::Mat W_samples(feat_cvmat.rows, feat_cvmat.rows, CV_32FC1, cvScalarAll(0.0));
			for (int i = 0; i < feat_cvmat.rows; ++i) {
				for (int k = 0; k < feat_cvmat.rows; ++k) {
					if (i == k)
						continue;
					W_samples.at<float>(i, k) = exp(-D_samples.at<float>(i, k) /** D_samples.at<float>(i, k) */ / sigma_square);
				}
			}
			// re-organize labels
			map<int, vector<int64> > label_indice_map;
			for (int i = 0; i < num_samples; ++i) {
				label_indice_map[opt_labels[i]].push_back(i);
			}
			vector<vector<int64> > label_indice;
			vector<int> labels;
			for (std::map<int, vector<int64>>::iterator it = label_indice_map.begin(); it != label_indice_map.end(); ++it) {
				labels.push_back(it->first);
				label_indice.push_back((it->second));
			}

			// optimize labels to reduce the loss
			OptimizeLabels(label_indice, W_samples, topK);
			num_categ_ = 0;
			// propagate label_indice to opt_labels
			for (int i = 0; i < label_indice.size(); ++i) {
				if (label_indice[i].size() > 0) {
					++num_categ_;
				}
				for (int k = 0; k < label_indice[i].size(); ++k) {
					opt_labels[label_indice[i][k]] = i; // labels[i];
				}
			}
		}
	}

	// added by Jianwei Yang @09/28/2015
	// main body for updating labels
	/*========================================*/
	/* y* = argmin_{y} L(W_{t-1}, y_{t-1}, y) */
	/*========================================*/
	/* to find optimal y, y*, we use an agglom*/
	/* erative clustering algorithm on the fea*/
	/* tures extracted from certian layer.    */
	/*=========================================*/
	template <typename Dtype>
	void Solver<Dtype>::UpdateLabels(int idx_start, int idx_end) {
		idx_start = idx_start == start_pos ? 0 : idx_start;
		idx_end = idx_end == end_pos ? num_ - 1 : idx_end;
		if (epoch_ == 0) { // if it is the first epoch_, then we must initialize labels for all samples
			idx_start = 0;
			idx_end = num_ - 1;
		}

		int* labels_pre = new int[num_];
		if (epoch_ >= skip_order_) { // if it is not the first epoch_, we inherit labels from previous epoches
			memcpy(labels_pre, label_pre_epoches_[epoch_ - skip_order_], num_ * sizeof(int));
		}
		label_pre_epoches_.push_back(labels_pre);

		// extract features from samples range in idx_start to idx_end
		float* features = ExtFeatures(idx_start, idx_end); // we extract features of all samples, and then choose the specified section 
		cv::Mat features_cvmat = cv::Mat((idx_end - idx_start + 1), dim_feature_, CV_32FC1, features, 0);
		
		// centralize
		//cv::Mat features_cvmat_mean;
		//cv::reduce(features_cvmat, features_cvmat_mean, 0, CV_REDUCE_AVG);
		//for (int i = 0; i < features_cvmat.rows; ++i) {
		//	features_cvmat.row(i) -= features_cvmat_mean;
		//}
		// features_cvmat = features_cvmat - features_cvmat_mean;
		// save features
		//if (!is_final_eval_) {
		//	h_writebmat_.write_bmat(dbpath_ + string("repo/feature_epoch_") + ::to_string(epoch_) + string(".bmat"),
		//		features, (idx_end - idx_start + 1), dim_feature_, string("float"), 0, 1);
		//}
		//else {
		//	h_writebmat_.write_bmat(dbpath_ + string("repo/feature_final_epoch_") + ::to_string(epoch_) + string(".bmat"),
		//		features, (idx_end - idx_start + 1), dim_feature_, string("float"), 0, 1);
		//}
		// L2 normalization on features
		if (do_l2norm_) {
			for (int i = 0; i < features_cvmat.rows; ++i) {
				features_cvmat.row(i) = features_cvmat.row(i) / (cv::norm(features_cvmat.row(i), cv::NORM_L2) + 1e-8);
			}
		}

		if (!is_final_eval_) {
			if (with_local_contrast_) {
				h_writebmat_.write_bmat(dbpath_ + string("/repo/feature_cvmat_lcc_epoch_") + ::to_string(features_cvmat.rows) + string("_") + ::to_string(epoch_) + string(".bmat"),
					features_cvmat.data, features_cvmat.rows, features_cvmat.cols, string("float"), 0, 1);
			}
			else {
				h_writebmat_.write_bmat(dbpath_ + string("/repo/feature_cvmat_epoch_") + ::to_string(features_cvmat.rows) + string("_") + ::to_string(epoch_) + string(".bmat"),
					features_cvmat.data, features_cvmat.rows, features_cvmat.cols, string("float"), 0, 1);
			}
		}
		else { // ::to_string(epoch_) + 
			if (with_local_contrast_) {
				h_writebmat_.write_bmat(dbpath_ + string("/repo/feature_final_lcc_cvmat_epoch_") + ::to_string(features_cvmat.rows) + string("_") + ::to_string(K_s_0_) + "_" + ::to_string(K_c_) + "_" + ::to_string(ratio_stage_) + string(".bmat"),
					features_cvmat.data, features_cvmat.rows, features_cvmat.cols, string("float"), 0, 1);
			}
			else {
				h_writebmat_.write_bmat(dbpath_ + string("/repo/feature_final_cvmat_epoch_") + ::to_string(features_cvmat.rows) + string("_") + ::to_string(K_s_0_) + "_" + ::to_string(K_c_) + "_" + ::to_string(ratio_stage_) + string(".bmat"),
					features_cvmat.data, features_cvmat.rows, features_cvmat.cols, string("float"), 0, 1);
			}

		}

		while (label_gt_ == NULL) {
			label_gt_ = test_nets_[0]->GetDataGTLabels();
			h_writebmat_.write_bmat(dbpath_ + string("/repo/labels_gt_") + ::to_string(features_cvmat.rows) + string(".bmat"),
				(unsigned char*)label_gt_, features_cvmat.rows, 1, string("int"), 0, 1);
		}

		// find optimal labels for data given features
		FindOptimalLables(features_cvmat, features, K_s_, 1, 1, label_pre_epoches_[epoch_] + idx_start);
		// FindOptimalLables_noKnn(features_cvmat, features, K_s_, 1, 1, label_pre_epoches_[epoch_] + idx_start);

		// save predicted labels
		if (!is_final_eval_) {
			if (with_local_contrast_) {
				h_writebmat_.write_bmat(dbpath_ + string("/repo/label_pre_lcc_epoch_") + ::to_string(features_cvmat.rows) + string("_") + ::to_string(epoch_) + string(".bmat"),
					(unsigned char*)label_pre_epoches_[epoch_], num_, 1, string("int"), 0, 1);
			}
			else {
				h_writebmat_.write_bmat(dbpath_ + string("/repo/label_pre_epoch_") + ::to_string(features_cvmat.rows) + string("_") + ::to_string(epoch_) + string(".bmat"),
					(unsigned char*)label_pre_epoches_[epoch_], num_, 1, string("int"), 0, 1);
			}
		}
		else {
			if (with_local_contrast_) {
				h_writebmat_.write_bmat(dbpath_ + string("/repo/label_final_pre_lcc_epoch_") + ::to_string(features_cvmat.rows) + string("_") + ::to_string(K_s_0_) + "_" + ::to_string(K_c_) + "_" + ::to_string(ratio_stage_) + string(".bmat"),
					(unsigned char*)label_pre_epoches_[epoch_], num_, 1, string("int"), 0, 1);
			}
			else {
				h_writebmat_.write_bmat(dbpath_ + string("/repo/label_final_pre_epoch_") + ::to_string(features_cvmat.rows) + string("_") + ::to_string(K_s_0_) + "_" + ::to_string(K_c_) + "_" + ::to_string(ratio_stage_) + string(".bmat"),
					(unsigned char*)label_pre_epoches_[epoch_], num_, 1, string("int"), 0, 1);
			}
		}

		// caculate the nmi 
		// float nmi = measure_nmi_ ? measure_.m_nmi(label_gt_, label_pre_epoches_[epoch_], num_) : 0;
		float nmi = measure_nmi_ ? measure_.m_nmi_fast(label_gt_, label_pre_epoches_[epoch_], num_) : 0;
		LOG(INFO) << "NCat " << num_categ_ << ": NMI = " << nmi;

		// propagate predicted labels to the data layer
		//if (epoch_ == 0) {
			if (!net_->SetDataLabels(label_pre_epoches_[epoch_], 0, num_ - 1)) {
				LOG(INFO) << "Reset Data Labels Failed";
			}
		//}
	}

	// added by Jianwei Yang @ 09/28/2015
	// main body for updating model parameters
	/*========================================*/
	/* W* = argmin_{W} L(W, W_{t-1}, y_{t-1}) */
	/*========================================*/
	template <typename Dtype>
	void Solver<Dtype>::UpdateModels(int iters) {
		vector<Blob<Dtype>*> bottom_vec;
		const int start_iter = iter_;
		const int stop_iter = iter_ + iters;
		int average_loss = this->param_.average_loss();
		vector<Dtype> losses;
		Dtype smoothed_loss = 0;

		// because there are batches prefetched, but their labels are from previous epoch,
		// we forward several times before trainig using forwardbackward
		int num_mini_batches = std::ceil(double(num_) / double(mini_batch_size_train_));
		for (int i = 0; i < num_mini_batches; ++i)
			net_->ForwardFromTo(0, 1);

		// net_->reset();
		// after updating labels, we then update the dimension of softmax layer
		// and then preturb the model first for robusty

		// change dimension of softmax layer

		// perturb network

		// f (epoch_ >= skip_order_)
			// ApplyPreturb();

		while (iter_ < stop_iter) {
			// zero-init the params
			net_->ClearParamDiffs();

			if (param_.test_interval() && iter_ % param_.test_interval() == 0
				&& (iter_ > 0 || param_.test_initialization())
				&& Caffe::root_solver()) {
				TestAll();
				if (requested_early_exit_) {
					// Break out of the while loop because stop was requested while testing.
					break;
				}
			}

			for (int i = 0; i < callbacks_.size(); ++i) {
				callbacks_[i]->on_start();
			}
			const bool display = param_.display() && iter_ % param_.display() == 0;
			net_->set_debug_info(display && param_.debug_info());
			// accumulate the loss and gradient
			Dtype loss = 0;
			for (int i = 0; i < param_.iter_size(); ++i) {
				loss += net_->ForwardBackward(bottom_vec);
			}

			loss /= param_.iter_size();
			// average the loss across iterations for smoothed reporting
			if (losses.size() < average_loss) {
				losses.push_back(loss);
				int size = losses.size();
				smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
			}
			else {
				int idx = (iter_ - start_iter) % average_loss;
				smoothed_loss += (loss - losses[idx]) / average_loss;
				losses[idx] = loss;
			}
			if (display) {
				LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
					<< ", loss = " << smoothed_loss;
				const vector<Blob<Dtype>*>& result = net_->output_blobs();
				int score_index = 0;
				for (int j = 0; j < result.size(); ++j) {
					const Dtype* result_vec = result[j]->cpu_data();
					const string& output_name =
						net_->blob_names()[net_->output_blob_indices()[j]];
					const Dtype loss_weight =
						net_->blob_loss_weights()[net_->output_blob_indices()[j]];
					for (int k = 0; k < result[j]->count(); ++k) {
						ostringstream loss_msg_stream;
						if (loss_weight) {
							loss_msg_stream << " (* " << loss_weight
								<< " = " << loss_weight * result_vec[k] << " loss)";
						}
						LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
							<< score_index++ << ": " << output_name << " = "
							<< result_vec[k] << loss_msg_stream.str();
					}
				}
			}
			for (int i = 0; i < callbacks_.size(); ++i) {
				callbacks_[i]->on_gradients_ready();
			}
			ApplyUpdate();
			
			// Increment the internal iter_ counter -- its value should always indicate
			// the number of times the weights have been updated.
			++iter_;

			SolverAction::Enum request = GetRequestedAction();

			// Save a snapshot if needed.
			if ((param_.snapshot()
				&& iter_ % param_.snapshot() == 0
				&& Caffe::root_solver()) ||
				(request == SolverAction::SNAPSHOT)) {
				Snapshot();
			}
			if (SolverAction::STOP == request) {
				requested_early_exit_ = true;
				// Break out of training loop.
				break;
			}

			/// added by Jianwei Yang @ 10/02/2015
			// when the loss is smaller than 1.0, then suspend the updating of CNN
			// if (loss < 1.0)
				// break;
		}

		// share weights to test net
		CHECK_NOTNULL(test_nets_[0].get())->
			ShareTrainedLayersWith(net_.get());
	}
	// added by Jianwei Yang @ 09/28/2015
	// main body for unsupervised learning

	// added by Jianwei Yang @ 09/28/2015
	// main body for assessing the suspend condition
	template <typename Dtype>
	bool Solver<Dtype>::Time2Suspend() {
		if (num_categ_ <= num_final_class_)
			return true;
		return false;
	}

	template <typename Dtype>
	void Solver<Dtype>::Step_Unsup(int iters) {
		// initialize epoch
		epoch_ = 0;
		K_NN_ = 20;
		// get training data size
		num_train_ = net_->GetDataSize();
		// get testing data size
		num_test_ = test_nets_[0]->GetDataSize();
		assert(num_train_ == num_test_)("train data size must be equal to test data size");
		num_ = num_train_;
		// get mini_batch_size for training
		mini_batch_size_train_ = net_->GetMiniBatchSize();
		// get mini_batch_size for testing
		mini_batch_size_test_ = test_nets_[0]->GetMiniBatchSize();

		// initialize label_gt
		label_gt_ = NULL;
		with_local_contrast_ = true;
		is_final_eval_ = false;

		if (ac_lcc_) {
			iters = 0;
			ratio_stage_ = 0;
		}
		/*=============================================*/
		/*Joint Learning of model parameters and labels*/
		/*using coordinate descent to update parameters*/
		/*and labels iteratively.                      */
		/*added by Jianwei Yang @09/28/2015            */
		/*=============================================*/


		while (1) {
			// initialize feature layer name
			// get feature dimension
			// get the dimesion of each feature layer
			// assess the suspend criterion
			// update labels for training data by fixing model parameters
			UpdateLabels(start_pos, end_pos);

			if (Time2Suspend()) {
				break;
			}
			// update model parameters by fixing data labels
			UpdateModels(iters);



			// self-add epoch
			++epoch_;

			// reduce learning iterations
			// iters = iters * 0.8;

		}
		/*===============================================*/
		// after learning the final feature representation
		// conduct agglomerateive clustering with local co
		// -ntrast
		/*===============================================*/
		is_final_eval_ = true;
		epoch_ = 0;
		label_pre_epoches_.clear();
		while (1) {
			UpdateLabels(start_pos, end_pos);
			++epoch_;
			if (Time2Suspend()) {
				break;
			}
		}

		/*===============================================*/
		// after learning the final feature representation
		// conduct agglomerateive clustering with local co
		// -ntrast
		/*===============================================*/
		//is_final_eval_ = true;
		//with_local_contrast_ = false;
		//epoch_ = 0;
		//label_pre_epoches_.clear();
		//while (1) {
		//	UpdateLabels(start_pos, end_pos);
		//	++epoch_;
		//	if (Time2Suspend()) {
		//		break;
		//	}
		//}
	}

	// added by Jianwei Yang @ 09/28/2015
	// main body for solving unsueprvised learning method
	template <typename Dtype>
	void Solver<Dtype>::Solve_Unsup(const char* resume_file) {
		CHECK(Caffe::root_solver());
		LOG(INFO) << "Solving " << net_->name();
		LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

		// Initialize to false every time we start solving.
		requested_early_exit_ = false;

		if (resume_file) {
			LOG(INFO) << "Restoring previous solver status from " << resume_file;
			Restore(resume_file);
		}

		// For a network that is trained by the solver, no bottom or top vecs
		// should be given, and we will just provide dummy vecs.
		Step_Unsup(param_.max_iter() - iter_);
		// If we haven't already, save a snapshot after optimization, unless
		// overridden by setting snapshot_after_train := false
		if (param_.snapshot_after_train()
			&& (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
			Snapshot();
		}
		if (requested_early_exit_) {
			LOG(INFO) << "Optimization stopped early.";
			return;
		}
		// After the optimization is done, run an additional train and test pass to
		// display the train and test loss/outputs if appropriate (based on the
		// display and test_interval settings, respectively).  Unlike in the rest of
		// training, for the train net we only run a forward pass as we've already
		// updated the parameters "max_iter" times -- this final pass is only done to
		// display the loss, which is computed in the forward pass.
		if (param_.display() && iter_ % param_.display() == 0) {
			Dtype loss;
			net_->ForwardPrefilled(&loss);
			LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
		}
		if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
			TestAll();
		}
		LOG(INFO) << "Optimization Done.";
	}

	template <typename Dtype>
	void Solver<Dtype>::TestAll() {
		for (int test_net_id = 0;
			test_net_id < test_nets_.size() && !requested_early_exit_;
			++test_net_id) {
			Test(test_net_id);
		}
	}

	template <typename Dtype>
	void Solver<Dtype>::Test(const int test_net_id) {
		CHECK(Caffe::root_solver());
		LOG(INFO) << "Iteration " << iter_
			<< ", Testing net (#" << test_net_id << ")";
		CHECK_NOTNULL(test_nets_[test_net_id].get())->
			ShareTrainedLayersWith(net_.get());
		vector<Dtype> test_score;
		vector<int> test_score_output_id;
		vector<Blob<Dtype>*> bottom_vec;
		const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
		Dtype loss = 0;
		for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
			SolverAction::Enum request = GetRequestedAction();
			// Check to see if stoppage of testing/training has been requested.
			while (request != SolverAction::NONE) {
				if (SolverAction::SNAPSHOT == request) {
					Snapshot();
				}
				else if (SolverAction::STOP == request) {
					requested_early_exit_ = true;
				}
				request = GetRequestedAction();
			}
			if (requested_early_exit_) {
				// break out of test loop.
				break;
			}

			Dtype iter_loss;
			const vector<Blob<Dtype>*>& result =
				test_net->Forward(bottom_vec, &iter_loss);
			if (param_.test_compute_loss()) {
				loss += iter_loss;
			}
			if (i == 0) {
				for (int j = 0; j < result.size(); ++j) {
					const Dtype* result_vec = result[j]->cpu_data();
					for (int k = 0; k < result[j]->count(); ++k) {
						test_score.push_back(result_vec[k]);
						test_score_output_id.push_back(j);
					}
				}
			}
			else {
				int idx = 0;
				for (int j = 0; j < result.size(); ++j) {
					const Dtype* result_vec = result[j]->cpu_data();
					for (int k = 0; k < result[j]->count(); ++k) {
						test_score[idx++] += result_vec[k];
					}
				}
			}
		}
		if (requested_early_exit_) {
			LOG(INFO) << "Test interrupted.";
			return;
		}
		if (param_.test_compute_loss()) {
			loss /= param_.test_iter(test_net_id);
			LOG(INFO) << "Test loss: " << loss;
		}
		for (int i = 0; i < test_score.size(); ++i) {
			const int output_blob_index =
				test_net->output_blob_indices()[test_score_output_id[i]];
			const string& output_name = test_net->blob_names()[output_blob_index];
			const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
			ostringstream loss_msg_stream;
			const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
			if (loss_weight) {
				loss_msg_stream << " (* " << loss_weight
					<< " = " << loss_weight * mean_score << " loss)";
			}
			LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
				<< mean_score << loss_msg_stream.str();
		}
	}

	template <typename Dtype>
	void Solver<Dtype>::Snapshot() {
		CHECK(Caffe::root_solver());
		string model_filename;
		switch (param_.snapshot_format()) {
		case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
			model_filename = SnapshotToBinaryProto();
			break;
		case caffe::SolverParameter_SnapshotFormat_HDF5:
			model_filename = SnapshotToHDF5();
			break;
		default:
			LOG(FATAL) << "Unsupported snapshot format.";
		}

		SnapshotSolverState(model_filename);
	}

	template <typename Dtype>
	string Solver<Dtype>::SnapshotFilename(const string extension) {
		string filename(param_.snapshot_prefix());
		const int kBufferSize = 20;
		char iter_str_buffer[kBufferSize];
		sprintf_s(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
		return filename + iter_str_buffer + extension;
	}

	template <typename Dtype>
	string Solver<Dtype>::SnapshotToBinaryProto() {
		string model_filename = SnapshotFilename(".caffemodel");
		LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
		NetParameter net_param;
		net_->ToProto(&net_param, param_.snapshot_diff());
		WriteProtoToBinaryFile(net_param, model_filename);
		return model_filename;
	}

	template <typename Dtype>
	string Solver<Dtype>::SnapshotToHDF5() {
		string model_filename = SnapshotFilename(".caffemodel.h5");
		LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
		net_->ToHDF5(model_filename, param_.snapshot_diff());
		return model_filename;
	}

	template <typename Dtype>
	void Solver<Dtype>::Restore(const char* state_file) {
		CHECK(Caffe::root_solver());
		string state_filename(state_file);
		if (state_filename.size() >= 3 &&
			state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
			RestoreSolverStateFromHDF5(state_filename);
		}
		else {
			RestoreSolverStateFromBinaryProto(state_filename);
		}
	}

	// Return the current learning rate. The currently implemented learning rate
	// policies are as follows:
	//    - fixed: always return base_lr.
	//    - step: return base_lr * gamma ^ (floor(iter / step))
	//    - exp: return base_lr * gamma ^ iter
	//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
	//    - multistep: similar to step but it allows non uniform steps defined by
	//      stepvalue
	//    - poly: the effective learning rate follows a polynomial decay, to be
	//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
	//    - sigmoid: the effective learning rate follows a sigmod decay
	//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
	//
	// where base_lr, max_iter, gamma, step, stepvalue and power are defined
	// in the solver parameter protocol buffer, and iter is the current iteration.
	template <typename Dtype>
	Dtype SGDSolver<Dtype>::GetLearningRate() {
		Dtype rate;
		const string& lr_policy = this->param_.lr_policy();
		if (lr_policy == "fixed") {
			rate = this->param_.base_lr();
		}
		else if (lr_policy == "step") {
			this->current_step_ = this->iter_ / this->param_.stepsize();
			rate = this->param_.base_lr() *
				pow(this->param_.gamma(), this->current_step_);
		}
		else if (lr_policy == "exp") {
			rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
		}
		else if (lr_policy == "inv") {
			rate = this->param_.base_lr() *
				pow(Dtype(1) + this->param_.gamma() * this->iter_,
				-this->param_.power());
		}
		else if (lr_policy == "multistep") {
			if (this->current_step_ < this->param_.stepvalue_size() &&
				this->iter_ >= this->param_.stepvalue(this->current_step_)) {
				this->current_step_++;
				LOG(INFO) << "MultiStep Status: Iteration " <<
					this->iter_ << ", step = " << this->current_step_;
			}
			rate = this->param_.base_lr() *
				pow(this->param_.gamma(), this->current_step_);
		}
		else if (lr_policy == "poly") {
			rate = this->param_.base_lr() * pow(Dtype(1.) -
				(Dtype(this->iter_) / Dtype(this->param_.max_iter())),
				this->param_.power());
		}
		else if (lr_policy == "sigmoid") {
			rate = this->param_.base_lr() * (Dtype(1.) /
				(Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
				Dtype(this->param_.stepsize())))));
		}
		else {
			LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
		}
		return rate;
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::PreSolve() {
		// Initialize the history
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		history_.clear();
		update_.clear();
		temp_.clear();
		for (int i = 0; i < net_params.size(); ++i) {
			const vector<int>& shape = net_params[i]->shape();
			history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
			update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
			temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
		}
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::ClipGradients() {
		const Dtype clip_gradients = this->param_.clip_gradients();
		if (clip_gradients < 0) { return; }
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		Dtype sumsq_diff = 0;
		for (int i = 0; i < net_params.size(); ++i) {
			sumsq_diff += net_params[i]->sumsq_diff();
		}
		const Dtype l2norm_diff = std::sqrt(sumsq_diff);
		if (l2norm_diff > clip_gradients) {
			Dtype scale_factor = clip_gradients / l2norm_diff;
			LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
				<< l2norm_diff << " > " << clip_gradients << ") "
				<< "by scale factor " << scale_factor;
			for (int i = 0; i < net_params.size(); ++i) {
				net_params[i]->scale_diff(scale_factor);
			}
		}
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::ApplyUpdate() {
		CHECK(Caffe::root_solver());
		Dtype rate = GetLearningRate();
		if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
			LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
		}
		ClipGradients();
		for (int param_id = 0; param_id < this->net_->learnable_params().size();
			++param_id) {
			Normalize(param_id);
			Regularize(param_id);
			ComputeUpdateValue(param_id, rate);
		}
		this->net_->Update();
	}

	/*===========================*/
	/* preturb network parameters*/
	/* added by Jianwei Yang     */
	/*===========================*/
	template <typename Dtype>
	void SGDSolver<Dtype>::ApplyPreturb() {
		CHECK(Caffe::root_solver());
		Dtype rate = GetLearningRate();
		if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
			LOG(INFO) << "Preturb Parameters @ Epoch " << this->epoch_ << ", lr = " << rate;
		}
		for (int param_id = 0; param_id < this->net_->learnable_params().size();
			++param_id) {
			AddNoises(param_id, 0.01);
		}
		this->net_->Update();
	}

	/*=================================*/
	/* add noises to network parameters*/
	/* added by Jianwei Yang */
	/* scale: scale of noises*
	/* ntype: noise type, e.g., uniform*/
	/*=================================*/
	template <typename Dtype>
	void SGDSolver<Dtype>::AddNoises(int param_id, Dtype scale) {
		// Scale gradient to counterbalance accumulation.
		const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
		const vector<Blob<Dtype>*>& net_learnable_params = this->net_->learnable_params();
		const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
		switch (Caffe::mode()) {
		case Caffe::CPU: {
			// caffe_scal(net_params[param_id]->count(), accum_normalization,
				// net_params[param_id]->mutable_cpu_diff());
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			// generate noises			
			//caffe_gpu_rng_gaussian(net_learnable_params[param_id]->count(), Dtype(0.0), scale,
			//	net_learnable_params[param_id]->mutable_gpu_data());
			caffe_gpu_rng_gaussian(net_learnable_params[param_id]->count(), Dtype(0.0), scale,
				net_learnable_params[param_id]->mutable_gpu_diff());
			// caffe_gpu_scal(net_learnable_params[param_id]->count(), accum_normalization,
				// net_learnable_params[param_id]->mutable_gpu_diff());
#else
			NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::Normalize(int param_id) {
		if (this->param_.iter_size() == 1) { return; }
		// Scale gradient to counterbalance accumulation.
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
		switch (Caffe::mode()) {
		case Caffe::CPU: {
			caffe_scal(net_params[param_id]->count(), accum_normalization,
				net_params[param_id]->mutable_cpu_diff());
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
				net_params[param_id]->mutable_gpu_diff());
#else
			NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::Regularize(int param_id) {
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		const vector<float>& net_params_weight_decay =
			this->net_->params_weight_decay();
		Dtype weight_decay = this->param_.weight_decay();
		string regularization_type = this->param_.regularization_type();
		Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
		switch (Caffe::mode()) {
		case Caffe::CPU: {
			if (local_decay) {
				if (regularization_type == "L2") {
					// add weight decay
					caffe_axpy(net_params[param_id]->count(),
						local_decay,
						net_params[param_id]->cpu_data(),
						net_params[param_id]->mutable_cpu_diff());
				}
				else if (regularization_type == "L1") {
					caffe_cpu_sign(net_params[param_id]->count(),
						net_params[param_id]->cpu_data(),
						temp_[param_id]->mutable_cpu_data());
					caffe_axpy(net_params[param_id]->count(),
						local_decay,
						temp_[param_id]->cpu_data(),
						net_params[param_id]->mutable_cpu_diff());
				}
				else {
					LOG(FATAL) << "Unknown regularization type: " << regularization_type;
				}
			}
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			if (local_decay) {
				if (regularization_type == "L2") {
					// add weight decay
					caffe_gpu_axpy(net_params[param_id]->count(),
						local_decay,
						net_params[param_id]->gpu_data(),
						net_params[param_id]->mutable_gpu_diff());
				}
				else if (regularization_type == "L1") {
					caffe_gpu_sign(net_params[param_id]->count(),
						net_params[param_id]->gpu_data(),
						temp_[param_id]->mutable_gpu_data());
					caffe_gpu_axpy(net_params[param_id]->count(),
						local_decay,
						temp_[param_id]->gpu_data(),
						net_params[param_id]->mutable_gpu_diff());
				}
				else {
					LOG(FATAL) << "Unknown regularization type: " << regularization_type;
				}
			}
#else
			NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		const vector<float>& net_params_lr = this->net_->params_lr();
		Dtype momentum = this->param_.momentum();
		Dtype local_rate = rate * net_params_lr[param_id];
		// Compute the update to history, then copy it to the parameter diff.
		switch (Caffe::mode()) {
		case Caffe::CPU: {
			caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
				net_params[param_id]->cpu_diff(), momentum,
				history_[param_id]->mutable_cpu_data());
			caffe_copy(net_params[param_id]->count(),
				history_[param_id]->cpu_data(),
				net_params[param_id]->mutable_cpu_diff());
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
				net_params[param_id]->gpu_diff(), momentum,
				history_[param_id]->mutable_gpu_data());
			caffe_copy(net_params[param_id]->count(),
				history_[param_id]->gpu_data(),
				net_params[param_id]->mutable_gpu_diff());
#else
			NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
		switch (this->param_.snapshot_format()) {
		case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
			SnapshotSolverStateToBinaryProto(model_filename);
			break;
		case caffe::SolverParameter_SnapshotFormat_HDF5:
			SnapshotSolverStateToHDF5(model_filename);
			break;
		default:
			LOG(FATAL) << "Unsupported snapshot format.";
		}
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
		const string& model_filename) {
		SolverState state;
		state.set_iter(this->iter_);
		state.set_learned_net(model_filename);
		state.set_current_step(this->current_step_);
		state.clear_history();
		for (int i = 0; i < history_.size(); ++i) {
			// Add history
			BlobProto* history_blob = state.add_history();
			history_[i]->ToProto(history_blob);
		}
		string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
		LOG(INFO)
			<< "Snapshotting solver state to binary proto file" << snapshot_filename;
		WriteProtoToBinaryFile(state, snapshot_filename.c_str());
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
		const string& model_filename) {
		string snapshot_filename =
			Solver<Dtype>::SnapshotFilename(".solverstate.h5");
		LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
		hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
			H5P_DEFAULT, H5P_DEFAULT);
		CHECK_GE(file_hid, 0)
			<< "Couldn't open " << snapshot_filename << " to save solver state.";
		hdf5_save_int(file_hid, "iter", this->iter_);
		hdf5_save_string(file_hid, "learned_net", model_filename);
		hdf5_save_int(file_hid, "current_step", this->current_step_);
		hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
			H5P_DEFAULT);
		CHECK_GE(history_hid, 0)
			<< "Error saving solver state to " << snapshot_filename << ".";
		for (int i = 0; i < history_.size(); ++i) {
			ostringstream oss;
			oss << i;
			hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
		}
		H5Gclose(history_hid);
		H5Fclose(file_hid);
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
		const string& state_file) {
		SolverState state;
		ReadProtoFromBinaryFile(state_file, &state);
		this->iter_ = state.iter();
		if (state.has_learned_net()) {
			NetParameter net_param;
			ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
			this->net_->CopyTrainedLayersFrom(net_param);
		}
		this->current_step_ = state.current_step();
		CHECK_EQ(state.history_size(), history_.size())
			<< "Incorrect length of history blobs.";
		LOG(INFO) << "SGDSolver: restoring history";
		for (int i = 0; i < history_.size(); ++i) {
			history_[i]->FromProto(state.history(i));
		}
	}

	template <typename Dtype>
	void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
		hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
		this->iter_ = hdf5_load_int(file_hid, "iter");
		if (H5LTfind_dataset(file_hid, "learned_net")) {
			string learned_net = hdf5_load_string(file_hid, "learned_net");
			this->net_->CopyTrainedLayersFrom(learned_net);
		}
		this->current_step_ = hdf5_load_int(file_hid, "current_step");
		hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
		CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
		int state_history_size = hdf5_get_num_links(history_hid);
		CHECK_EQ(state_history_size, history_.size())
			<< "Incorrect length of history blobs.";
		for (int i = 0; i < history_.size(); ++i) {
			ostringstream oss;
			oss << i;
			hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
				kMaxBlobAxes, history_[i].get());
		}
		H5Gclose(history_hid);
		H5Fclose(file_hid);
	}

	template <typename Dtype>
	void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
		CHECK(Caffe::root_solver());
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		const vector<float>& net_params_lr = this->net_->params_lr();
		Dtype momentum = this->param_.momentum();
		Dtype local_rate = rate * net_params_lr[param_id];
		switch (Caffe::mode()) {
		case Caffe::CPU: {
			// save history momentum for stepping back
			caffe_copy(net_params[param_id]->count(),
				this->history_[param_id]->cpu_data(),
				this->update_[param_id]->mutable_cpu_data());

			// update history
			caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
				net_params[param_id]->cpu_diff(), momentum,
				this->history_[param_id]->mutable_cpu_data());

			// compute update: step back then over step
			caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
				this->history_[param_id]->cpu_data(), -momentum,
				this->update_[param_id]->mutable_cpu_data());

			// copy
			caffe_copy(net_params[param_id]->count(),
				this->update_[param_id]->cpu_data(),
				net_params[param_id]->mutable_cpu_diff());
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			// save history momentum for stepping back
			caffe_copy(net_params[param_id]->count(),
				this->history_[param_id]->gpu_data(),
				this->update_[param_id]->mutable_gpu_data());

			// update history
			caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
				net_params[param_id]->gpu_diff(), momentum,
				this->history_[param_id]->mutable_gpu_data());

			// compute update: step back then over step
			caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
				this->history_[param_id]->gpu_data(), -momentum,
				this->update_[param_id]->mutable_gpu_data());

			// copy
			caffe_copy(net_params[param_id]->count(),
				this->update_[param_id]->gpu_data(),
				net_params[param_id]->mutable_gpu_diff());
#else
			NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void AdaGradSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
		CHECK(Caffe::root_solver());
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		const vector<float>& net_params_lr = this->net_->params_lr();
		Dtype delta = this->param_.delta();
		Dtype local_rate = rate * net_params_lr[param_id];
		switch (Caffe::mode()) {
		case Caffe::CPU: {
			// compute square of gradient in update
			caffe_powx(net_params[param_id]->count(),
				net_params[param_id]->cpu_diff(), Dtype(2),
				this->update_[param_id]->mutable_cpu_data());

			// update history
			caffe_add(net_params[param_id]->count(),
				this->update_[param_id]->cpu_data(),
				this->history_[param_id]->cpu_data(),
				this->history_[param_id]->mutable_cpu_data());

			// prepare update
			caffe_powx(net_params[param_id]->count(),
				this->history_[param_id]->cpu_data(), Dtype(0.5),
				this->update_[param_id]->mutable_cpu_data());

			caffe_add_scalar(net_params[param_id]->count(),
				delta, this->update_[param_id]->mutable_cpu_data());

			caffe_div(net_params[param_id]->count(),
				net_params[param_id]->cpu_diff(),
				this->update_[param_id]->cpu_data(),
				this->update_[param_id]->mutable_cpu_data());

			// scale and copy
			caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
				this->update_[param_id]->cpu_data(), Dtype(0),
				net_params[param_id]->mutable_cpu_diff());
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			// compute square of gradient in update
			caffe_gpu_powx(net_params[param_id]->count(),
				net_params[param_id]->gpu_diff(), Dtype(2),
				this->update_[param_id]->mutable_gpu_data());

			// update history
			caffe_gpu_add(net_params[param_id]->count(),
				this->update_[param_id]->gpu_data(),
				this->history_[param_id]->gpu_data(),
				this->history_[param_id]->mutable_gpu_data());

			// prepare update
			caffe_gpu_powx(net_params[param_id]->count(),
				this->history_[param_id]->gpu_data(), Dtype(0.5),
				this->update_[param_id]->mutable_gpu_data());

			caffe_gpu_add_scalar(net_params[param_id]->count(),
				delta, this->update_[param_id]->mutable_gpu_data());

			caffe_gpu_div(net_params[param_id]->count(),
				net_params[param_id]->gpu_diff(),
				this->update_[param_id]->gpu_data(),
				this->update_[param_id]->mutable_gpu_data());

			// scale and copy
			caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
				this->update_[param_id]->gpu_data(), Dtype(0),
				net_params[param_id]->mutable_gpu_diff());
#else
			NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		const vector<float>& net_params_lr = this->net_->params_lr();

		// get the learning rate
		Dtype delta = this->param_.delta();
		Dtype rms_decay = this->param_.rms_decay();
		Dtype local_rate = rate * net_params_lr[param_id];

		switch (Caffe::mode()) {
		case Caffe::CPU:
			// compute square of gradient in update
			caffe_powx(net_params[param_id]->count(),
				net_params[param_id]->cpu_diff(), Dtype(2),
				this->update_[param_id]->mutable_cpu_data());

			// update history
			caffe_cpu_axpby(net_params[param_id]->count(),
				Dtype(1 - rms_decay), this->update_[param_id]->cpu_data(),
				rms_decay, this->history_[param_id]->mutable_cpu_data());

			// prepare update
			caffe_powx(net_params[param_id]->count(),
				this->history_[param_id]->cpu_data(), Dtype(0.5),
				this->update_[param_id]->mutable_cpu_data());

			caffe_add_scalar(net_params[param_id]->count(),
				delta, this->update_[param_id]->mutable_cpu_data());

			caffe_div(net_params[param_id]->count(),
				net_params[param_id]->cpu_diff(), this->update_[param_id]->cpu_data(),
				this->update_[param_id]->mutable_cpu_data());

			// scale and copy
			caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
				this->update_[param_id]->cpu_data(), Dtype(0),
				net_params[param_id]->mutable_cpu_diff());
			break;
		case Caffe::GPU:
#ifndef CPU_ONLY
			// compute square of gradient in update
			caffe_gpu_powx(net_params[param_id]->count(),
				net_params[param_id]->gpu_diff(), Dtype(2),
				this->update_[param_id]->mutable_gpu_data());

			// update history
			caffe_gpu_axpby(net_params[param_id]->count(),
				Dtype(1 - rms_decay), this->update_[param_id]->gpu_data(),
				rms_decay, this->history_[param_id]->mutable_gpu_data());

			// prepare update
			caffe_gpu_powx(net_params[param_id]->count(),
				this->history_[param_id]->gpu_data(), Dtype(0.5),
				this->update_[param_id]->mutable_gpu_data());

			caffe_gpu_add_scalar(net_params[param_id]->count(),
				delta, this->update_[param_id]->mutable_gpu_data());

			caffe_gpu_div(net_params[param_id]->count(),
				net_params[param_id]->gpu_diff(), this->update_[param_id]->gpu_data(),
				this->update_[param_id]->mutable_gpu_data());

			caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
				this->update_[param_id]->gpu_data(), Dtype(0),
				net_params[param_id]->mutable_gpu_diff());
#else
			NO_GPU;
#endif
			break;
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void AdaDeltaSolver<Dtype>::AdaDeltaPreSolve() {
		// Add the extra history entries for AdaDelta after those from
		// SGDSolver::PreSolve
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		for (int i = 0; i < net_params.size(); ++i) {
			const vector<int>& shape = net_params[i]->shape();
			this->history_.push_back(
				shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
		}
	}

	template <typename Dtype>
	void AdaDeltaSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		const vector<float>& net_params_lr = this->net_->params_lr();
		Dtype delta = this->param_.delta();
		Dtype momentum = this->param_.momentum();
		Dtype local_rate = rate * net_params_lr[param_id];
		size_t update_history_offset = net_params.size();
		switch (Caffe::mode()) {
		case Caffe::CPU: {
			// compute square of gradient in update
			caffe_powx(net_params[param_id]->count(),
				net_params[param_id]->cpu_diff(), Dtype(2),
				this->update_[param_id]->mutable_cpu_data());

			// update history of gradients
			caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
				this->update_[param_id]->cpu_data(), momentum,
				this->history_[param_id]->mutable_cpu_data());

			// add delta to history to guard against dividing by zero later
			caffe_set(net_params[param_id]->count(), delta,
				this->temp_[param_id]->mutable_cpu_data());

			caffe_add(net_params[param_id]->count(),
				this->temp_[param_id]->cpu_data(),
				this->history_[update_history_offset + param_id]->cpu_data(),
				this->update_[param_id]->mutable_cpu_data());

			caffe_add(net_params[param_id]->count(),
				this->temp_[param_id]->cpu_data(),
				this->history_[param_id]->cpu_data(),
				this->temp_[param_id]->mutable_cpu_data());

			// divide history of updates by history of gradients
			caffe_div(net_params[param_id]->count(),
				this->update_[param_id]->cpu_data(),
				this->temp_[param_id]->cpu_data(),
				this->update_[param_id]->mutable_cpu_data());

			// jointly compute the RMS of both for update and gradient history
			caffe_powx(net_params[param_id]->count(),
				this->update_[param_id]->cpu_data(), Dtype(0.5),
				this->update_[param_id]->mutable_cpu_data());

			// compute the update
			caffe_mul(net_params[param_id]->count(),
				net_params[param_id]->cpu_diff(),
				this->update_[param_id]->cpu_data(),
				net_params[param_id]->mutable_cpu_diff());

			// compute square of update
			caffe_powx(net_params[param_id]->count(),
				net_params[param_id]->cpu_diff(), Dtype(2),
				this->update_[param_id]->mutable_cpu_data());

			// update history of updates
			caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
				this->update_[param_id]->cpu_data(), momentum,
				this->history_[update_history_offset + param_id]->mutable_cpu_data());

			// apply learning rate
			caffe_cpu_scale(net_params[param_id]->count(), local_rate,
				net_params[param_id]->cpu_diff(),
				net_params[param_id]->mutable_cpu_diff());
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			// compute square of gradient in update
			caffe_gpu_powx(net_params[param_id]->count(),
				net_params[param_id]->gpu_diff(), Dtype(2),
				this->update_[param_id]->mutable_gpu_data());

			// update history of gradients
			caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
				this->update_[param_id]->gpu_data(), momentum,
				this->history_[param_id]->mutable_gpu_data());

			// add delta to history to guard against dividing by zero later
			caffe_gpu_set(net_params[param_id]->count(), delta,
				this->temp_[param_id]->mutable_gpu_data());

			caffe_gpu_add(net_params[param_id]->count(),
				this->temp_[param_id]->gpu_data(),
				this->history_[update_history_offset + param_id]->gpu_data(),
				this->update_[param_id]->mutable_gpu_data());

			caffe_gpu_add(net_params[param_id]->count(),
				this->temp_[param_id]->gpu_data(),
				this->history_[param_id]->gpu_data(),
				this->temp_[param_id]->mutable_gpu_data());

			// divide history of updates by history of gradients
			caffe_gpu_div(net_params[param_id]->count(),
				this->update_[param_id]->gpu_data(),
				this->temp_[param_id]->gpu_data(),
				this->update_[param_id]->mutable_gpu_data());

			// jointly compute the RMS of both for update and gradient history
			caffe_gpu_powx(net_params[param_id]->count(),
				this->update_[param_id]->gpu_data(), Dtype(0.5),
				this->update_[param_id]->mutable_gpu_data());

			// compute the update and copy to net_diff
			caffe_gpu_mul(net_params[param_id]->count(),
				net_params[param_id]->gpu_diff(),
				this->update_[param_id]->gpu_data(),
				net_params[param_id]->mutable_gpu_diff());

			// compute square of update
			caffe_gpu_powx(net_params[param_id]->count(),
				net_params[param_id]->gpu_diff(), Dtype(2),
				this->update_[param_id]->mutable_gpu_data());

			// update history of updates
			caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) - momentum,
				this->update_[param_id]->gpu_data(), momentum,
				this->history_[update_history_offset + param_id]->mutable_gpu_data());

			// apply learning rate
			caffe_gpu_scale(net_params[param_id]->count(), local_rate,
				net_params[param_id]->gpu_diff(),
				net_params[param_id]->mutable_gpu_diff());
#else
			NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	template <typename Dtype>
	void AdamSolver<Dtype>::AdamPreSolve() {
		// Add the extra history entries for Adam after those from
		// SGDSolver::PreSolve
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		for (int i = 0; i < net_params.size(); ++i) {
			const vector<int>& shape = net_params[i]->shape();
			this->history_.push_back(
				shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
		}
	}

	template <typename Dtype>
	void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		const vector<float>& net_params_lr = this->net_->params_lr();
		Dtype local_rate = rate * net_params_lr[param_id];
		const Dtype beta1 = this->param_.momentum();
		const Dtype beta2 = this->param_.momentum2();

		// we create aliases for convenience
		size_t update_history_offset = net_params.size();
		Blob<Dtype>* val_m = this->history_[param_id].get();
		Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
		Blob<Dtype>* val_t = this->temp_[param_id].get();

		const int t = this->iter_ + 1;
		const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
			(Dtype(1.) - pow(beta1, t));
		const int N = net_params[param_id]->count();
		const Dtype eps_hat = this->param_.delta();

		switch (Caffe::mode()) {
		case Caffe::CPU: {
			// update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
			caffe_cpu_axpby(N, Dtype(1) - beta1,
				net_params[param_id]->cpu_diff(), beta1,
				val_m->mutable_cpu_data());

			// update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
			caffe_mul(N,
				net_params[param_id]->cpu_diff(),
				net_params[param_id]->cpu_diff(),
				val_t->mutable_cpu_data());
			caffe_cpu_axpby(N, Dtype(1) - beta2,
				val_t->cpu_data(), beta2,
				val_v->mutable_cpu_data());

			// set update
			caffe_powx(N,
				val_v->cpu_data(), Dtype(0.5),
				val_t->mutable_cpu_data());
			caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
			caffe_div(N,
				val_m->cpu_data(),
				val_t->cpu_data(),
				val_t->mutable_cpu_data());

			caffe_cpu_scale(N, local_rate*correction,
				val_t->cpu_data(),
				net_params[param_id]->mutable_cpu_diff());
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			// update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
			caffe_gpu_axpby(N, Dtype(1) - beta1,
				net_params[param_id]->gpu_diff(), beta1,
				val_m->mutable_gpu_data());

			// update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
			caffe_gpu_mul(N,
				net_params[param_id]->gpu_diff(),
				net_params[param_id]->gpu_diff(),
				val_t->mutable_gpu_data());
			caffe_gpu_axpby(N, Dtype(1) - beta2,
				val_t->gpu_data(), beta2,
				val_v->mutable_gpu_data());

			// set update
			caffe_gpu_powx(N,
				val_v->gpu_data(), Dtype(0.5),
				val_t->mutable_gpu_data());
			caffe_gpu_add_scalar(N, eps_hat,
				val_t->mutable_gpu_data());
			caffe_gpu_div(N,
				val_m->gpu_data(),
				val_t->gpu_data(),
				val_t->mutable_gpu_data());

			caffe_gpu_scale(N, local_rate*correction,
				val_t->gpu_data(),
				net_params[param_id]->mutable_gpu_diff());
#else
			NO_GPU;
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}

	INSTANTIATE_CLASS(Solver);
	INSTANTIATE_CLASS(SGDSolver);
	INSTANTIATE_CLASS(NesterovSolver);
	INSTANTIATE_CLASS(AdaGradSolver);
	INSTANTIATE_CLASS(RMSPropSolver);
	INSTANTIATE_CLASS(AdaDeltaSolver);
	INSTANTIATE_CLASS(AdamSolver);

}  // namespace caffe
