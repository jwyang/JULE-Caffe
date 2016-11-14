#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void GraphLinkageLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// get the nunmber of samples
		//int num = bottom[0]->num();
		//int count = bottom[0]->count();
		//const int channels = bottom[0]->channels();

		//const Dtype* features = bottom[0]->cpu_data();
		//const Dtype* labels = bottom[1]->cpu_data();
		//// compute distance matrix
		//sigma_ = 0.0;
		//for (int i = 0; i < num; ++i) {
		//	for (int j = 0; j < num; ++j) {
		//		caffe_sub(channels,
		//			bottom[0]->cpu_data() + (i * channels),
		//			bottom[0]->cpu_data() + (j * channels),
		//			diff_.mutable_cpu_data() + (i * num + j) * channels);
		//		Dtype d_sq = caffe_cpu_dot(channels,
		//			diff_.cpu_data() + (i * num + j) * channels,
		//			diff_.cpu_data() + (i * num + j) * channels);

		//		dist_sq_.mutable_cpu_data()[i * num + j] = d_sq;
		//		sigma_ += d_sq;
		//	}
		//}

		//sigma_ /= num * num;

		//for (int i = 0; i < num; ++i) {
		//	for (int j = 0; j < num; ++j) {
		//		Dtype d_sq = dist_sq_.cpu_data()[i * num + j];
		//		// if (d_sq < sigma_ * sigma_) {
		//		if (1) {
		//			affinity_sample_.mutable_cpu_data()[i * num + j] = exp(-d_sq / sigma_);
		//		}
		//		else {
		//			affinity_sample_.mutable_cpu_data()[i * num + j] = 0;
		//		}
		//	}
		//}

		//// build label map
		//std::map<int, std::vector<int> > label_indice_map;
		//for (int i = 0; i < num; ++i) {
		//	label_indice_map[labels[i]].push_back(i);
		//}

		//// convert label map to
		//vector<int> labels_idx;
		//vector<vector<int> > label_indice;
		//for (std::map<int, vector<int>>::iterator it = label_indice_map.begin(); it != label_indice_map.end(); ++it) {
		//	labels_idx.push_back(it->first);
		//	label_indice.push_back(it->second);
		//}
		//label_indice_ = label_indice;
		////// compute the loss = loss_intra + loss_extra
		//affinity_intra_.Reshape(label_indice.size(), 1, 1, 1);
		///// compute intra loss loss_intra
		//for (int i = 0; i < label_indice.size(); ++i) {
		//	Dtype val(0.0);
		//	//for (int m = 0; m < label_indice[i].size(); ++m) {
		//	//	for (int n = 0; n < label_indice[i].size(); ++n) {
		//	//		Dtype entry_m_n = affinity_sample_.cpu_data()[label_indice[i][m] * num +
		//	//			label_indice[i][n]];
		//	//		Dtype entry_n_m = affinity_sample_.cpu_data()[label_indice[i][n] * num +
		//	//			label_indice[i][m]];
		//	//		val += entry_m_n * entry_n_m;
		//	//	}
		//	//}
		//	for (int m = 0; m < label_indice[i].size(); ++m) {
		//		for (int n = 0; n < label_indice[i].size(); ++n) {
		//			if (m == n)
		//				continue;
		//			Dtype entry_m_n = affinity_sample_.cpu_data()[label_indice[i][m] * num +
		//				label_indice[i][n]];
		//			val += entry_m_n;
		//		}
		//	}

		//	affinity_intra_.mutable_cpu_data()[i] = val;
		//	// loss_intra += 1 - val / label_indice[i].size() / label_indice[i].size();
		//}

		///// compute extra loss loss_extra
		//affinity_extra_.Reshape(label_indice.size() * label_indice.size(), 1, 1, 1);

		//for (int i = 0; i < label_indice.size(); ++i) {
		//	for (int j = 0; j < label_indice.size(); ++j) {
		//		if (i == j) {
		//			affinity_extra_.mutable_cpu_data()[j * label_indice.size() + i] = 0;
		//			continue;
		//		}
		//		Dtype A_c_i_j = 0;

		//		//for (int m = 0; m < label_indice[i].size(); ++m) {

		//		//	Dtype s_W_c_j_i = 0;
		//		//	for (int n = 0; n < label_indice[j].size(); ++n) {
		//		//		s_W_c_j_i += affinity_sample_.cpu_data()[label_indice[j][n] * num +
		//		//			label_indice[i][m]];
		//		//		// W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
		//		//	}

		//		//	Dtype s_W_c_i_j = 0;
		//		//	for (int n = 0; n < label_indice[j].size(); ++n) {
		//		//		s_W_c_i_j += affinity_sample_.cpu_data()[label_indice[i][m] * num +
		//		//			label_indice[j][n]];
		//		//		// W_samples.at<float>(label_indice[i][m], label_indice[j][n]);
		//		//	}

		//		//	A_c_i_j += s_W_c_j_i * s_W_c_i_j;
		//		//}

		//		for (int m = 0; m < label_indice[i].size(); ++m) {

		//			Dtype s_W_c_j_i = 0;
		//			for (int n = 0; n < label_indice[j].size(); ++n) {
		//				s_W_c_j_i += affinity_sample_.cpu_data()[label_indice[j][n] * num +
		//					label_indice[i][m]];
		//				// W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
		//			}
		//			A_c_i_j += s_W_c_j_i;
		//		}

		//		affinity_extra_.mutable_cpu_data()[j * label_indice.size() + i] = A_c_i_j;

		//	}
		//}
		//Dtype loss_intra(0.0), loss_extra(0.0);
		//Dtype loss(0.0);

		//int num_intra_valid = 0;
		//for (int i = 0; i < label_indice.size(); ++i) {
		//	if (label_indice[i].size() == 1)
		//		continue;
		//	loss_intra += 1 - affinity_intra_.cpu_data()[i] / label_indice[i].size() / (label_indice[i].size() - 1);
		//	++num_intra_valid;
		//}
		//num_intra_valid_ = num_intra_valid;
		//for (int i = 0; i < label_indice.size(); ++i) {
		//	for (int j = 0; j < label_indice.size(); ++j) {
		//		loss_extra += affinity_extra_.cpu_data()[i * label_indice.size() + j]
		//			/ label_indice[i].size() / label_indice[j].size();  // A_c_i_j
		//	}
		//}

		//loss = loss_intra / num_intra_valid + loss_extra / label_indice.size() / label_indice.size();
		//top[0]->mutable_cpu_data()[0] = loss;

		/*===============================================================*/
		/* contrastive loss                                              */
		/*===============================================================*/
		//// get the nunmber of samples
		//int num = bottom[0]->num();
		//int count = bottom[0]->count();
		//const int channels = bottom[0]->channels();

		//const Dtype* features = bottom[0]->cpu_data();
		//const Dtype* labels = bottom[1]->cpu_data();
		//// compute distance matrix
		//sigma_ = 0.0;
		//for (int i = 0; i < num; ++i) {
		//	for (int j = 0; j < num; ++j) {
		//		//Dtype norm_i = caffe_cpu_dot(channels,
		//		//	bottom[0]->cpu_data() + (i * channels),
		//		//	bottom[0]->cpu_data() + (i * channels));
		//		//Dtype norm_j = caffe_cpu_dot(channels,
		//		//	bottom[0]->cpu_data() + (j * channels),
		//		//	bottom[0]->cpu_data() + (j * channels));

		//		//caffe_scal(channels, Dtype(1.0) / sqrt(norm_i), bottom[0]->mutable_cpu_data() + (i * channels));
		//		//caffe_scal(channels, Dtype(1.0) / sqrt(norm_j), bottom[0]->mutable_cpu_data() + (j * channels));

		//		caffe_sub(channels,
		//			bottom[0]->cpu_data() + (i * channels),
		//			bottom[0]->cpu_data() + (j * channels),
		//			diff_.mutable_cpu_data() + (i * num + j) * channels);

		//		Dtype d_sq = caffe_cpu_dot(channels,
		//			diff_.cpu_data() + (i * num + j) * channels,
		//			diff_.cpu_data() + (i * num + j) * channels);

		//		dist_sq_.mutable_cpu_data()[i * num + j] = d_sq;
		//		sigma_ += d_sq;
		//	}
		//}

		//sigma_ = sigma_ / (num * num);

		//for (int i = 0; i < num; ++i) {
		//	for (int j = 0; j < num; ++j) {
		//		Dtype d_sq = dist_sq_.cpu_data()[i * num + j];
		//		// if (d_sq < sigma_ * sigma_) {
		//		if (1) {
		//			affinity_sample_.mutable_cpu_data()[i * num + j] = exp(-d_sq / sigma_);
		//		}
		//		else {
		//			affinity_sample_.mutable_cpu_data()[i * num + j] = 0;
		//		}
		//	}
		//}

		//// build label map
		//std::map<int, std::vector<int> > label_indice_map;
		//for (int i = 0; i < num; ++i) {
		//	label_indice_map[labels[i]].push_back(i);
		//}

		//// convert label map to
		//vector<int> labels_idx;
		//vector<vector<int> > label_indice;
		//for (std::map<int, vector<int>>::iterator it = label_indice_map.begin(); it != label_indice_map.end(); ++it) {
		//	labels_idx.push_back(it->first);
		//	label_indice.push_back(it->second);
		//}
		//label_indice_ = label_indice;
		////// compute the loss = loss_intra + loss_extra
		//Dtype loss_intra(0.0), loss_extra(0.0);
		//Dtype loss(0.0);

		//int num_intra_valid = 0;
		//int num_extra_valid = 0;
		//for (int i = 0; i < num; ++i) {
		//	for (int j = 0; j < num; ++j) {
		//		if (i == j)
		//			continue;
		//		if (labels[i] == labels[j]) { // intra pairs
		//			++num_intra_valid;
		//			loss_intra += (1 - affinity_sample_.cpu_data()[i * num + j]);
		//		}
		//		else if (labels[i] != labels[j]) { // extra pairs
		//			++num_extra_valid;
		//			loss_extra += affinity_sample_.cpu_data()[i * num + j] > 0.3 ? 
		//				affinity_sample_.cpu_data()[i * num + j] - 0.3: 0;
		//		}
		//	}
		//}
		//num_intra_valid_ = num_intra_valid;
		//num_extra_valid_ = num_extra_valid;
		//if (num_intra_valid == 0) {
		//	loss = loss_extra / num_extra_valid;
		//}
		//else {
		//	loss = loss_intra / num_intra_valid + loss_extra / num_extra_valid;
		//}
		//top[0]->mutable_cpu_data()[0] = loss;

		/*======================================================================*/
		/* triplet loss                                                         */
		/*======================================================================*/
		// get the nunmber of samples
		int num = bottom[0]->num();
		int count = bottom[0]->count();
		const int channels = bottom[0]->channels();

		const Dtype* features = bottom[0]->cpu_data();
		const Dtype* labels = bottom[1]->cpu_data();

		// build label map
		std::map<int, std::vector<int> > label_indice_map;
		for (int i = 0; i < num; ++i) {
			label_indice_map[labels[i]].push_back(i);
		}

		// convert label map to
		vector<int> labels_idx;
		vector<vector<int> > label_indice;
		for (std::map<int, vector<int>>::iterator it = label_indice_map.begin(); it != label_indice_map.end(); ++it) {
			labels_idx.push_back(it->first);
			label_indice.push_back(it->second);
		}
		label_indice_ = label_indice;
		num_items_ = 0;
		// compute number of items
		int num_neg_sampling = label_indice.size() - 1 > 10 ? 10 : label_indice.size() - 1;
		for (int i = 0; i < label_indice.size(); ++i) {
			if (label_indice[i].size() > 1) { // 
				num_items_ += label_indice[i].size() * (label_indice[i].size() - 1) * num_neg_sampling;
			}
		}

		if (num_items_ == 0) {
			top[0]->mutable_cpu_data()[0] = 0;
			return;
		}
		diff_pos_.Reshape(num_items_, channels, 1, 1);
		diff_neg_.Reshape(num_items_, channels, 1, 1);

		rec_pos_.clear();
		rec_neg_.clear();
		int id_items = 0;
		// compute triplet loss
		for (int i = 0; i < label_indice.size(); ++i) {
			if (label_indice[i].size() > 1) { // 
				for (int m = 0; m < label_indice[i].size(); ++m) {
					for (int n = 0; n < label_indice[i].size(); ++n) {
						if (m == n)
							continue;
						int idx_m = label_indice[i][m];
						int idx_n = label_indice[i][n];
						// compute diffs
						vector<bool> is_choosed(num, false);
						while (1) {
							int idx = rand() % num;
							if (!is_choosed[idx] && labels[idx] != labels[idx_m]) {
								// compute extra diff
								caffe_sub(channels,
									bottom[0]->cpu_data() + (idx_m * channels),
									bottom[0]->cpu_data() + (idx_n * channels),
									diff_pos_.mutable_cpu_data() + id_items * channels);

								// compute intra diff
								caffe_sub(channels,
									bottom[0]->cpu_data() + (idx_m * channels),
									bottom[0]->cpu_data() + (idx * channels),
									diff_neg_.mutable_cpu_data() + id_items * channels);

								rec_pos_.push_back(make_pair(idx_m, idx_n));
								rec_neg_.push_back(make_pair(idx_m, idx));
								is_choosed[idx] = true;
								++id_items;
							}

							if (id_items % num_neg_sampling == 0)
								break;
						}
					}
				}
			}
		}
		num_items_ = id_items;
		Dtype loss(0.0);
		dist_sq_pos_.Reshape(num_items_, channels, 1, 1);
		dist_sq_neg_.Reshape(num_items_, channels, 1, 1);
		dist_sq_.Reshape(num_items_, 1, 1, 1);
		for (int i = 0; i < num_items_; ++i) {
			// Triplet loss accumulation
			// Loss component calculated from a and b
			dist_sq_pos_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
				diff_pos_.cpu_data() + (i*channels), diff_pos_.cpu_data() + (i*channels));
			// a b is a similar pair for triplet
			dist_sq_.mutable_cpu_data()[i] = dist_sq_pos_.cpu_data()[i];
			// Loss component calculated from a and c
			dist_sq_neg_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
				diff_neg_.cpu_data() + (i*channels), diff_neg_.cpu_data() + (i*channels));
			// a c is a dissimilar pair for triplet
			dist_sq_.mutable_cpu_data()[i] -= dist_sq_neg_.cpu_data()[i];
			loss += std::max(margin_ + dist_sq_.cpu_data()[i], Dtype(0.0));  // loss accumulated accumulated by the triplet part
		}
		loss = loss / num_items_ / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void GraphLinkageLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		// Dtype margin = this->layer_param_.contrastive_loss_param().margin();
		//if (propagate_down[1]) {
		//	LOG(FATAL) << this->type()
		//		<< " Layer cannot backpropagate to label inputs.";
		//}
		///*=========================================*/
		///*propagate error gradient to bottom layers*/
		///*=========================================*/
		//if (propagate_down[0]) { // 
		//	Dtype* bout = bottom[0]->mutable_cpu_diff();
		//	int num = bottom[0]->num();
		//	int channels = bottom[0]->channels();
		//	for (int i = 0; i < label_indice_.size(); ++i) { // intra error propagate
		//		if (label_indice_[i].size() == 1)
		//			continue;
		//		for (int m = 0; m < label_indice_[i].size(); ++m) {
		//			for (int n = 0; n < label_indice_[i].size(); ++n) {
		//				int idx_m = label_indice_[i][m];
		//				int idx_n = label_indice_[i][n];
		//				// Dtype alpha = (-affinity_sample_.mutable_cpu_data()[idx_m * num + idx_n])
		//					// / (-sigma_) / label_indice_[i].size() / (label_indice_[i].size() - 1) / num_intra_valid_;
		//				Dtype alpha = 1 / label_indice_[i].size() / (label_indice_[i].size() - 1) / num_intra_valid_;
		//				caffe_cpu_axpby(
		//					channels,
		//					alpha,
		//					diff_.cpu_data() + (idx_m * num + idx_n) * channels,
		//					Dtype(1.0),
		//					bout + (idx_m * channels));
		//				caffe_cpu_axpby(
		//					channels,
		//					-alpha,
		//					diff_.cpu_data() + (idx_m * num + idx_n) * channels,
		//					Dtype(1.0),
		//					bout + (idx_n * channels));
		//			}
		//		}
		//	}

		//	for (int i = 0; i < label_indice_.size(); ++i) {
		//		for (int j = 0; j < label_indice_.size(); ++j) {
		//			for (int m = 0; m < label_indice_[i].size(); ++m) {
		//				for (int n = 0; n < label_indice_[j].size(); ++n) {
		//					int idx_m = label_indice_[i][m];
		//					int idx_n = label_indice_[j][n];
		//					// Dtype alpha = (affinity_sample_.mutable_cpu_data()[idx_m * num + idx_n])
		//						// / (-sigma_) / label_indice_[i].size() / label_indice_[j].size() / label_indice_.size() / label_indice_.size();
		//					Dtype alpha = -1 / label_indice_[i].size() / label_indice_[j].size() / label_indice_.size() / label_indice_.size();
		//					caffe_cpu_axpby(
		//						channels,
		//						alpha,
		//						diff_.cpu_data() + (idx_m * num + idx_n) * channels,
		//						Dtype(1.0),
		//						bout + (idx_m * channels));
		//					caffe_cpu_axpby(
		//						channels,
		//						-alpha,
		//						diff_.cpu_data() + (idx_m * num + idx_n) * channels,
		//						Dtype(1.0),
		//						bout + (idx_n * channels));
		//				}
		//			}
		//		}
		//	}
		//}

		/*=========================================*/
		/* back propagate for contrastive loss     */
		/*=========================================*/
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		/*=========================================*/
		/*propagate error gradient to bottom layers*/
		/*=========================================*/
		//if (propagate_down[0]) { // 			
		//	Dtype* bout = bottom[0]->mutable_cpu_diff();
		//	const Dtype* labels = bottom[1]->cpu_data();
		//	int num = bottom[0]->num();
		//	int channels = bottom[0]->channels();
		//	for (int i = 0; i < num; ++i) {
		//		for (int j = 0; j < num; ++j) {
		//			if (i == j)
		//				continue;
		//			if (labels[i] == labels[j]) { // intra pairs
		//				Dtype alpha = 0; // 2 * affinity_sample_.cpu_data()[i * num + j] / num_intra_valid_;
		//				caffe_cpu_axpby(
		//					channels,
		//					alpha,
		//					diff_.cpu_data() + (i * num + j) * channels,
		//					Dtype(1.0),
		//					bout + (i * channels));
		//				caffe_cpu_axpby(
		//					channels,
		//					-alpha,
		//					diff_.cpu_data() + (i * num + j) * channels,
		//					Dtype(1.0),
		//					bout + (j * channels));
		//			}
		//			else if (labels[i] != labels[j]) { // extra pairs
		//				if (affinity_sample_.cpu_data()[i * num + j] > 0.3) {
		//					Dtype alpha = -2 * affinity_sample_.cpu_data()[i * num + j] / num_extra_valid_;
		//					caffe_cpu_axpby(
		//						channels,
		//						alpha,
		//						diff_.cpu_data() + (i * num + j) * channels,
		//						Dtype(1.0),
		//						bout + (i * channels));
		//					caffe_cpu_axpby(
		//						channels,
		//						-alpha,
		//						diff_.cpu_data() + (i * num + j) * channels,
		//						Dtype(1.0),
		//						bout + (j * channels));
		//				}
		//				// loss_extra += max(0, affinity_sample_.cpu_data()[i * num + j] - 0.5);
		//			}
		//		}
		//	}
		//}

		/*====================================*/
		/* back propagate for triplet loss    */
		/*====================================*/

		if (propagate_down[0]) {
			caffe_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
			const Dtype sign = 1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] /
				static_cast<Dtype>(num_items_);
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			Dtype* bout = bottom[0]->mutable_cpu_diff();
			for (int i = 0; i < num_items_; ++i) {
				//caffe_cpu_axpby(
				//	channels,
				//	alpha,
				//	diff_pos_.cpu_data() + (i * channels),
				//	Dtype(1.0),
				//	bout + (rec_pos_[i].first * channels));

				//caffe_cpu_axpby(
				//	channels,
				//	-alpha,
				//	diff_pos_.cpu_data() + (i * channels),
				//	Dtype(1.0),
				//	bout + (rec_pos_[i].second * channels));

				if (margin_ + dist_sq_.cpu_data()[i] > Dtype(0.0)) {
					// similar pairs
					caffe_cpu_axpby(
						channels,
						alpha,
						diff_pos_.cpu_data() + (i * channels),
						Dtype(1.0),
						bout + (rec_pos_[i].first * channels));

					caffe_cpu_axpby(
						channels,
						-alpha,
						diff_pos_.cpu_data() + (i * channels),
						Dtype(1.0),
						bout + (rec_pos_[i].second * channels));

					// dissimilar pairs
					caffe_cpu_axpby(
						channels,
						-alpha,
						diff_neg_.cpu_data() + (i * channels),
						Dtype(1.0),
						bout + (rec_neg_[i].first * channels));


					caffe_cpu_axpby(
						channels,
						alpha,
						diff_neg_.cpu_data() + (i * channels),
						Dtype(1.0),
						bout + (rec_neg_[i].second * channels));

				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(GraphLinkageLossLayer);

}  // namespace caffe
