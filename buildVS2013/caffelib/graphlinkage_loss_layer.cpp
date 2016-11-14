#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void GraphLinkageLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		diff_.Reshape(bottom[0]->num() * bottom[0]->num(), bottom[0]->channels(), 1, 1);
		diff_sq_.Reshape(bottom[0]->num() * bottom[0]->num(), bottom[0]->channels(), 1, 1);
		dist_sq_.Reshape(bottom[0]->num() * bottom[0]->num(), 1, 1, 1);
		affinity_sample_.Reshape(bottom[0]->num() * bottom[0]->num(), 1, 1, 1);
		// vector of ones used to sum along channels
		summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
		for (int i = 0; i < bottom[0]->channels(); ++i)
			summer_vec_.mutable_cpu_data()[i] = Dtype(1);
		sigma_ = this->layer_param_.graphlinkage_loss_param().sigma();
		margin_ = 0.1;
	}

	template <typename Dtype>
	void GraphLinkageLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// get the nunmber of samples
		int num = bottom[0]->num();
		int count = bottom[0]->count();
		const int channels = bottom[0]->channels();

		const Dtype* features = bottom[0]->cpu_data();
		const Dtype* labels = bottom[1]->cpu_data();
		// compute distance matrix
		sigma_ = 0.0;
		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < num; ++j) {
				caffe_sub(channels,
					bottom[0]->cpu_data() + (i * channels),
					bottom[0]->cpu_data() + (j * channels),
					diff_.mutable_cpu_data() + (i * num + j) * channels);
				Dtype d_sq = caffe_cpu_dot(channels,
					diff_.cpu_data() + (i * num + j) * channels,
					diff_.cpu_data() + (i * num + j) * channels);

				dist_sq_.mutable_cpu_data()[i * num + j] = d_sq;
				sigma_ += d_sq;
			}
		}

		sigma_ = sigma_ / (num * num);

		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < num; ++j) {
				Dtype d_sq = dist_sq_.cpu_data()[i * num + j];
				// if (d_sq < sigma_ * sigma_) {
				if (1) {
					affinity_sample_.mutable_cpu_data()[i * num + j] = exp(-d_sq / sigma_);
				}
				else {
					affinity_sample_.mutable_cpu_data()[i * num + j] = 0;
				}
			}
		}

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
		//// compute the loss = loss_intra + loss_extra
		Dtype loss_intra(0.0), loss_extra(0.0);
		Dtype loss(0.0);

		int num_intra_valid = 0;
		int num_extra_valid = 0;
		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < num; ++j) {
				if (i == j)
					continue;
				if (labels[i] == labels[j]) { // intra pairs
					++num_intra_valid;
					loss_intra += (1 - affinity_sample_.cpu_data()[i * num + j]);
				}
				else if (labels[i] != labels[j]) { // extra pairs
					++num_extra_valid;
					loss_extra += affinity_sample_.cpu_data()[i * num + j] > 0.5 ? affinity_sample_.cpu_data()[i * num + j] - 0.5 : 0;
				}
			}
		}
		num_intra_valid_ = num_intra_valid;
		num_extra_valid_ = num_extra_valid;
		loss = loss_intra / num_intra_valid + loss_extra / num_extra_valid;
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void GraphLinkageLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		/*=========================================*/
		/*propagate error gradient to bottom layers*/
		/*=========================================*/
		if (propagate_down[0]) { // 			
			Dtype* bout = bottom[0]->mutable_cpu_diff();
			const Dtype* labels = bottom[1]->cpu_data();
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			for (int i = 0; i < num; ++i) {
				for (int j = 0; j < num; ++j) {
					if (i == j)
						continue;
					if (labels[i] == labels[j]) { // intra pairs
						Dtype alpha = 2 * affinity_sample_.cpu_data()[i * num + j] / sigma_ / num_intra_valid_;
						caffe_cpu_axpby(
							channels,
							alpha,
							diff_.cpu_data() + (i * num + j) * channels,
							Dtype(1.0),
							bout + (i * channels));
						caffe_cpu_axpby(
							channels,
							-alpha,
							diff_.cpu_data() + (i * num + j) * channels,
							Dtype(1.0),
							bout + (j * channels));
					}
					else if (labels[i] != labels[j]) { // extra pairs
						if (affinity_sample_.cpu_data()[i * num + j] > 0.5) {
							Dtype alpha = -2 * affinity_sample_.cpu_data()[i * num + j] / sigma_ / num_extra_valid_;
							caffe_cpu_axpby(
								channels,
								alpha,
								diff_.cpu_data() + (i * num + j) * channels,
								Dtype(1.0),
								bout + (i * channels));
							caffe_cpu_axpby(
								channels,
								-alpha,
								diff_.cpu_data() + (i * num + j) * channels,
								Dtype(1.0),
								bout + (j * channels));
						}
						// loss_extra += max(0, affinity_sample_.cpu_data()[i * num + j] - 0.5);
					}
				}
			}
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(GraphLinkageLossLayer);
#endif

	INSTANTIATE_CLASS(GraphLinkageLossLayer);
	REGISTER_LAYER_CLASS(GraphLinkageLoss);

}  // namespace caffe
