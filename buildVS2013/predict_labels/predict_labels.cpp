#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;


int main(int argc, char** argv)
{

	LOG(INFO) << argv[0] << " [GPU] [Device ID]";

	//Setting CPU or GPU
	if (argc >= 2 && strcmp(argv[1], "GPU") == 0)
	{
		Caffe::set_mode(Caffe::GPU);
		int device_id = 0;
		if (argc == 3)
		{
			device_id = atoi(argv[2]);
		}
		Caffe::SetDevice(device_id);
		LOG(INFO) << "Using GPU #" << device_id;
	}
	else
	{
		LOG(INFO) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}

	// Load net
	// Assume you are in Caffe master directory
	Net<float> net(string("examples/prediction_example/prediction_example.prototxt"), caffe::TEST);

	// Load pre-trained net (binary proto)
	// Assume you are already trained the cifar10 example.
	net.CopyTrainedLayersFrom("D:/Kaggle@DL/VT-F15-ECE6504-HW2-1.0/2_finetuning-alexnet-wikiart-style/models/caffe_alexnet_train_iter_1110.caffemodel");

	float loss = 0.0;
	ifstream fin("D:/Kaggle@DL/VT-F15-ECE6504-HW2-1.0/2_finetuning-alexnet-wikiart-style/data/wikiart/test.txt");
	ofstream fout("predict_results.sln.csv");
	for (int t = 0; t < 20; ++t) {
		vector<Blob<float>*> results = net.ForwardPrefilled(&loss);
		const boost::shared_ptr<Blob<float> >& probLayer = net.blob_by_name("prob");
		const float* probs_out = probLayer->cpu_data();
		std::cout << "batch: " << t << std::endl;
		// get label for maximal pro
		for (int i = 0; i < probLayer->num(); ++i) {
			int max_label = 0;
			float max_value = 0;
			for (int k = 0; k < probLayer->channels(); ++k) {
				if (probs_out[i * probLayer->channels() + k] > max_value) {
					max_value = probs_out[i * probLayer->channels() + k];
					max_label = k;
				}
			}
			string imgpath;
			int label;
			fin >> imgpath >> label;
			imgpath = imgpath.substr(0, imgpath.length() - 4) + ",";
			fout << imgpath << max_label << endl ;
		}
	}
	fin.close();
	fout.close();
	//LOG(INFO) << "Result size: " << results.size();

	//// Log how many blobs were loaded
	//LOG(INFO) << "Blob size: " << net.input_blobs().size();


	//LOG(INFO) << "-------------";
	//LOG(INFO) << " prediction :  ";

	// Get probabilities
	const boost::shared_ptr<Blob<float> >& probLayer = net.blob_by_name("prob");
	const float* probs_out = probLayer->cpu_data();

	// get label for maximal pro
	for (int i = 0; i < probLayer->num(); ++i) {
		int max_label = 0;
		float max_value = 0;
		for (int k = 0; k < probLayer->channels(); ++k) {
			if (probs_out[i * probLayer->channels() + k] > max_value) {
				max_value = probs_out[i * probLayer->channels() + k];
				max_label = k;
			}
		}
	}
	

	// Get argmax results
	//const boost::shared_ptr<Blob<float> >& argmaxLayer = net.blob_by_name("argmax");

	//// Display results
	//LOG(INFO) << "---------------------------------------------------------------";
	//const float* argmaxs = argmaxLayer->cpu_data();
	//for (int i = 0; i < argmaxLayer->num(); i++)
	//{
	//	LOG(INFO) << "Pattern:" << i << " class:" << argmaxs[i*argmaxLayer->height() + 0] << " Prob=" << probs_out[i*probLayer->height() + 0];
	//}
	//LOG(INFO) << "-------------";


	return 0;
}
