#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

	template <typename Dtype>
	DataLayer<Dtype>::DataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param),
		reader_(param) {
	}

	template <typename Dtype>
	DataLayer<Dtype>::~DataLayer() {
		delete[]labels_gt_;
		delete[]labels_pre_;
		this->StopInternalThread();
	}

	template <typename Dtype>
	void DataLayer<Dtype>::init_bmat_reader() { // init data reader
		std::string dbpath = this->layer_param_.data_param().source();
		// replace ending part with bmat file
		std::string dbpath_trim = dbpath.substr(0, dbpath.find_last_of("/"));
		std::string dbpath_data = dbpath_trim + "train_data.bmat";
		std::string dbpath_data_label = dbpath_trim + "train_data_label.bmat";
		std::string dbpath_data_mean = dbpath_trim + "train_data_mean.bmat";
	}
		 
	template <typename Dtype>
	void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int batch_size = this->layer_param_.data_param().batch_size();
		// Read a data point, and use it to initialize the top blob.
		Datum& datum = *(reader_.full().peek());

		// Use data_transformer to infer the expected blob shape from datum.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
		this->transformed_data_.Reshape(top_shape);
		// Reshape top[0] and prefetch_data according to the batch_size.
		top_shape[0] = batch_size;
		top[0]->Reshape(top_shape);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(top_shape);
		}
		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();
		// label
		if (this->output_labels_) {
			vector<int> label_shape(1, batch_size);
			top[1]->Reshape(label_shape);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].label_.Reshape(label_shape);
			}
		}

		// initialize for unsupervised learning
		idx_train_ = 0;
		idx_test_ = 0;
		epoch_ = 0;
		num_ = reader_.get_db_size();
		mini_batch_size_ = batch_size;
		labels_gt_ = new int[num_];
		labels_pre_ = new int[num_];
		is_labels_set_ = false;

		init_bmat_reader();
	}

	// This function is called on prefetch thread
	template<typename Dtype>
	void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());

		// Reshape according to the first datum of each batch
		// on single input batches allows for inputs of varying dimension.
		const int batch_size = this->layer_param_.data_param().batch_size();
		Datum& datum = *(reader_.full().peek());
		// Use data_transformer to infer the expected blob shape from datum.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
		this->transformed_data_.Reshape(top_shape);
		// Reshape batch according to the batch_size.
		top_shape[0] = batch_size;
		batch->data_.Reshape(top_shape);

		Dtype* top_data = batch->data_.mutable_cpu_data();
		Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

		if (this->output_labels_) {
			top_label = batch->label_.mutable_cpu_data();
		}
		if (this->phase_ == TEST) {
			for (int item_id = 0; item_id < batch_size; ++item_id) {
				timer.Start();
				// get a datum
				Datum& datum = *(reader_.full().pop("Waiting for data"));
				read_time += timer.MicroSeconds();
				timer.Start();
				// Apply data transformations (mirror, scale, crop...)
				int offset = batch->data_.offset(item_id);
				this->transformed_data_.set_cpu_data(top_data + offset);
				this->data_transformer_->Transform(datum, &(this->transformed_data_));
				// Copy label.
				if (this->output_labels_) {
					// top_label[item_id] = datum.label();
					// instead of assign groundtruth labels, we use the predicted labels
					// set labels for data 
					top_label[item_id] = datum.label();
					// set labels_gt_, added by Jianwei Yang @ 09/28/2015
					if (epoch_ == 0)
						labels_gt_[idx_test_] = datum.label();
				}
				trans_time += timer.MicroSeconds();
				// self-add idx_, added by Jianwei Yang @09/28/2015
				++idx_test_;
				if (idx_test_ == num_) {
					idx_test_ = 0;
					++epoch_;
					// LOG(INFO) << "Finish one epoch.";
				}
				reader_.free().push(const_cast<Datum*>(&datum));
			}
		}
		else {
			// instead of organize batch with the same order as in db, we need to manually
			// organize data in some way helpful for unsupervised training
			int item_id = 0;

			while (1) {
				timer.Start();
				// get a datum
				Datum& datum = *(reader_.full().pop("Waiting for data"));
				// if (rand() % 2 == 0) { // skip current sample with 0.5 propability
				if (0) {
					++idx_train_;
					if (idx_train_ == num_) {
						idx_train_ = 0;
						++epoch_;
						// LOG(INFO) << "Finish one epoch.";
					}
					reader_.free().push(const_cast<Datum*>(&datum));
				}
				else {
					read_time += timer.MicroSeconds();
					timer.Start();
					// Apply data transformations (mirror, scale, crop...)
					int offset = batch->data_.offset(item_id);
					this->transformed_data_.set_cpu_data(top_data + offset);
					this->data_transformer_->Transform(datum, &(this->transformed_data_));
					// Copy label.
					if (this->output_labels_) {
						// top_label[item_id] = datum.label();
						// instead of assign groundtruth labels, we use the predicted labels
						// set labels for data 		
						assert(is_labels_set_)("labels must be set before unsupervised training");
						// int idx_rand = datum.float_data(0); // idx_train_; // 
						int idx_rand = idx_train_; // idx_train_; //
						if (is_labels_set_)
							top_label[item_id] = labels_pre_[idx_rand];
						// set labels_gt_, added by Jianwei Yang @ 09/28/2015
						// if (epoch_ == 0)
						// labels_gt_[idx_train_] = datum.label();
					}
					trans_time += timer.MicroSeconds();
					// self-add idx_, added by Jianwei Yang @09/28/2015
					++idx_train_;
					if (idx_train_ == num_) {
						idx_train_ = 0;
						++epoch_;
						// LOG(INFO) << "Finish one epoch.";
					}
					reader_.free().push(const_cast<Datum*>(&datum));
					++item_id;
					if (item_id == batch_size)
						break;
				}
			}
		}
		timer.Stop();
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	// This function is called to re-initialize labels for data, added by Jianwei @ 09/28/2015
	template<typename Dtype>
	bool DataLayer<Dtype>::set_data_labels(int* labels, int idx_start, int idx_end) {
		assert(idx_start >= 0 && idx_start < num_)("idx_start must be in [0, num_)");
		assert(idx_end >= 0 && idx_end < num_)("idx_end must be in [0, num_)");
		if (idx_start > idx_end) {
			return false;
		}
		else {
			
			memcpy(labels_pre_ + idx_start, labels + idx_start, (idx_end - idx_start + 1) * sizeof(int));
			// ReleaseBatches();
			is_labels_set_ = true;
			return true;
		}
	}

	// This function is called to set the position to seek for data, added by Jianwei @ 09/28/2015
	template<typename Dtype>
	bool DataLayer<Dtype>::set_data_pos(int pos) {
		assert(pos >= 0 && pos < num_)("pos must be in [0, num_)");
		return false;
	}

	// This function is called to set the position to seek for data, added by Jianwei @ 09/28/2015
	template<typename Dtype>
	int DataLayer<Dtype>::get_data_pos() {
		return idx_test_ - prefetch_full_.size() * mini_batch_size_;
	}

	// This function is called to set the position to seek for data, added by Jianwei @ 09/28/2015
	template<typename Dtype>
	int DataLayer<Dtype>::get_data_size() {
		return num_;
	}

	// This function is called to get the groundtruth labels for data in this layer, added by Jianwei @ 09/29/2015
	template<typename Dtype>
	int* DataLayer<Dtype>::get_data_gtlabels() {
		if (epoch_ == 0) {
			return NULL;
		}
		else {
			return labels_gt_;
		}
	}

	// This function is called to get the groundtruth labels for data in this layer, added by Jianwei @ 09/29/2015
	template<typename Dtype>
	int DataLayer<Dtype>::get_data_minibatch_size() {
		return mini_batch_size_;
	}


	INSTANTIATE_CLASS(DataLayer);
	REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
