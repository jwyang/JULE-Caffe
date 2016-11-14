#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>
#include <random>
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/proto/caffe.pb.h"


#define use_bmat

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

DataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::randperm(int length, std::vector<int>& indice, int groupsize, bool turnon) {
	// step-1: generate length number of random float numbers
	std::vector<std::pair<float, int> > num_map(length);
	random_device rd;   //
	mt19937 gen(rd());  // generator
	int length_rd = ceil(length / groupsize);
	uniform_real_distribution<float> dis(0, 1);
	for (int i = 0; i < length_rd; ++i) {
		if (!turnon) {
			num_map[i] = (std::make_pair(float(i) / length_rd, i));
		}
		else {
			num_map[i] = (std::make_pair(dis(gen), i));
		}
	}

	// step-2: sort generated random float numbers
	std::sort(num_map.begin(), num_map.end());

	// step-3: get the indice
	for (int i = 0; i < length_rd; ++i) {
		for (int k = 0; k < groupsize; ++k) {
			if ((i * groupsize + k) >= length)
				continue;
			indice[i * groupsize + k] = min(num_map[i].second * groupsize + k, length - 1);
		}
	}
	// std::random_shuffle(indice.begin(), indice.end());
}

void DataReader::Body::InternalThreadEntry() {
  shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  db->Open(param_.data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<shared_ptr<QueuePair> > qps;
  bmat bmat_parser;
  //// get db size
  db_size_ = 0;
  while (cursor->valid()) {
	  // go to the next iter
	  cursor->Next();
	  ++db_size_;
  }
  LOG(INFO) << "DB Size: " << db_size_;
  DLOG(INFO) << "Restarting data prefetching from start.";
  cursor->SeekToFirst();

#ifdef use_bmat
  std::string leveldbpath = param_.data_param().source();
  std::string dbpath = leveldbpath.substr(0, leveldbpath.find("_leveldb"));
  // read bmat from hard drvier, inlcuding train_data, train_data_label and train_data_mean.
  temp_data tdata;
  temp_data tdata_meta;
  temp_data tdata_mean;
  temp_data tdata_label;

  bmat_parser.read_bmat(dbpath + "_data.bmat", tdata, INT64_MAX);
  //bmat_parser.read_bmat(dbpath + "_data_mean.bmat", tdata_mean, INT64_MAX);
  bmat_parser.read_bmat(dbpath + "_data_label.bmat", tdata_label, INT64_MAX);

  // get db size
  db_size_ = tdata.rows;
  LOG(INFO) << "DB Size: " << tdata.rows;
  //bmat_parser.read_bmat(dbpath + "_data_meta.bmat", tdata_meta, INT64_MAX);
  //tdata_label.value_int = new int[tdata_label.cols];
  //char* addr = tdata_label.value;
  //for (long long i = 0; i < tdata_label.rows; ++i) {
	 // for (long long j = 0; j < tdata_label.cols; ++j) {
		//  __int32* v = (__int32*)addr;
		//  addr = addr + sizeof(int);
		//  tdata_label.value_int[i * tdata_label.cols + j] = *v;
	 // }
  //}

  //tdata_meta.value_int = new int[tdata_meta.rows * tdata_meta.cols];
  //addr = tdata_meta.value;
  //for (long long i = 0; i < tdata_meta.rows; ++i) {
	 // for (long long j = 0; j < tdata_meta.cols; ++j) {
		//  __int32* v = (__int32*)addr;
		//  addr = addr + sizeof(int);
		//  tdata_meta.value_int[i * tdata_meta.cols + j] = *v;
	 // }
  //}
  bool isTrainPhase = dbpath.find("test") == std::string::npos;

  vector<int> indice;
  indice.clear();
  indice.resize(tdata.rows);
  randperm(tdata.rows, indice, 1, false);
#endif

  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
	int t = 0;
#ifdef use_bmat
	for (int i = 0; i < solver_count; ++i) {
		shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
		read_one_bmat(cursor.get(), indice[t], &tdata, &tdata_meta, &tdata_label, qp.get());
		++t;
		qps.push_back(qp);
	}
#else
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      read_one(cursor.get(), qp.get());
      qps.push_back(qp);
    }
#endif

    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
#ifdef use_bmat
		  if (t >= tdata.rows) {
			  if (isTrainPhase)
				  randperm(tdata.rows, indice, 1, true);
			  t = 0;
		  }
		  read_one_bmat(cursor.get(), indice[t], &tdata, &tdata_meta, &tdata_label, qps[i].get());
		  ++t;
#else
         read_one(cursor.get(), qps[i].get());
#endif
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
  Datum* datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  datum->ParseFromString(cursor->value());
  qp->full_.push(datum);

  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

void DataReader::Body::read_one_bmat(db::Cursor* cursor, int idx, temp_data* data, temp_data* meta, temp_data* label, QueuePair* qp) {
	Datum* datum = qp->free_.pop();
	datum->ParseFromString(cursor->value());
	// TODO deserialize in-place instead of copy?
	//datum->set_channels(meta->value_int[0]);
	//datum->set_height(meta->value_int[1]);
	//datum->set_width(meta->value_int[2]);
	//
	//datum->clear_data();
	//datum->clear_float_data();

	//char *dat = new char[data->cols];
	//memcpy(dat, data->GetCharData(idx), data->cols);
	std::string buffer = std::string(data->value + idx * data->cols, data->cols);
	datum->set_data(buffer);
	datum->set_label(label->value_int[idx]);
	datum->add_float_data(idx);
	// datum->set_encoded(false);

	qp->full_.push(datum);

	// LOG(INFO) << datum->label() << " " << datum->channels() << " " << datum->width() << " " << datum->height();
	// go to the next iter
	//cursor->Next();
	//if (!cursor->valid()) {
	//	DLOG(INFO) << "Restarting data prefetching from start.";
	//	cursor->SeekToFirst();
	//}
}

}  // namespace caffe
