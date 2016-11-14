#include "caffe/util/measure.h"
#include <opencv2/opencv.hpp>
using namespace std;
measure::measure()
{
}


measure::~measure()
{
}

float measure::m_nmi(int* _label_anchor, int* _label_predict, int length) { // measure normalized mutual information given two label groups

	int* label_anchor = new int[length];
	int* label_predict = new int[length];

	memcpy(label_anchor, _label_anchor, length * sizeof(int));
	memcpy(label_predict, _label_predict, length * sizeof(int));

	// step-1: compute entropy for label_anchor
	// step-1-1: sort label_anchor in increase order
	int* pos_gt = new int[length];
	for (int i = 0; i < length; ++i) {
		pos_gt[i] = i;
	}
	for (int i = 0; i < length; ++i) {
		for (int j = i + 1; j < length; ++j) {
			if (label_anchor[i] > label_anchor[j]) {
				swap(label_anchor[i], label_anchor[j]);
				swap(pos_gt[i], pos_gt[j]);
			}
		}
	}

	int label_s = label_anchor[0];
	int pos = 1;
	vector<vector<int> > idx_classes(0);
	vector<int> idx_class(1, 0);
	for (int i = pos; i < length; ++i) {
		if (label_anchor[i] != label_s) {
			label_s = label_anchor[i];
			idx_classes.push_back(idx_class);
			idx_class.clear();
			idx_class.push_back(pos_gt[i]);
		}
		else {
			idx_class.push_back(pos_gt[i]);
		}
	}
	idx_classes.push_back(idx_class);

	float ent_anchor = 0;
	for (int i = 0; i < idx_classes.size(); ++i) {
		float p = log(float(idx_classes[i].size()) / float(length));
		ent_anchor += idx_classes[i].size() * p;
	}

	// step-2: compute entropy for label_predicted
	int* pos_pred = new int[length];
	for (int i = 0; i < length; ++i) {
		pos_pred[i] = i;
	}

	for (int i = 0; i < length; ++i) {
		for (int j = i + 1; j < length; ++j) {
			if (label_predict[i] > label_predict[j]) {
				swap(label_predict[i], label_predict[j]);
				swap(pos_pred[i], pos_pred[j]);
			}
		}
	}

	label_s = label_predict[0];
	pos = 1;
	vector<vector<int> > idxp_classes(0);
	vector<int> idxp_class(1, 0);
	for (int i = pos; i < length; ++i) {
		if (label_predict[i] != label_s) {
			label_s = label_predict[i];
			idxp_classes.push_back(idxp_class);
			idxp_class.clear();
			idxp_class.push_back(pos_pred[i]);
		}
		else {
			idxp_class.push_back(pos_pred[i]);
		}
	}
	idxp_classes.push_back(idxp_class);

	float ent_predict = 0;
	for (int i = 0; i < idxp_classes.size(); ++i) {
		ent_predict += idxp_classes[i].size() * log(float(idxp_classes[i].size()) / float(length));
	}

	// step-3: compute mutual entropy between label_anchor and label_predicted
	float ment = 0;
	for (int i = 0; i < idx_classes.size(); ++i) {
		int n_i = idx_classes[i].size();
		for (int j = 0; j < idxp_classes.size(); ++j) {
			int n_j = idxp_classes[j].size();
			// count n_h,l
			int n_h_l = 0;
			for (int m = 0; m < idx_classes[i].size(); ++m) {
				for (int n = 0; n < idxp_classes[j].size(); ++n) {
					if (idx_classes[i][m] == idxp_classes[j][n])
						++n_h_l;
				}
			}

			// calculation
			if (n_h_l == 0)
				continue;

			ment += n_h_l * log(float(length * n_h_l) / float(n_i * n_j));
		}
	}

	delete[] pos_gt;
	delete[] pos_pred;
	delete[] label_anchor;
	delete[] label_predict;
	return ment / sqrt(ent_anchor * ent_predict);

}

float measure::m_nmi_fast(int* _label_anchor, int* _label_predict, int length) { // measure normalized mutual information given two label groups

	int* label_anchor = new int[length];
	int* label_predict = new int[length];
	
	cv::Mat Manc, Mpred;

	memcpy(label_anchor, _label_anchor, length * sizeof(int));
	memcpy(label_predict, _label_predict, length * sizeof(int));

	// step-1: compute entropy for label_anchor
	// step-1-1: sort label_anchor in increase order
	int* pos_gt = new int[length];
	for (int i = 0; i < length; ++i) {
		pos_gt[i] = i;
	}
	for (int i = 0; i < length; ++i) {
		for (int j = i + 1; j < length; ++j) {
			if (label_anchor[i] > label_anchor[j]) {
				swap(label_anchor[i], label_anchor[j]);
				swap(pos_gt[i], pos_gt[j]);
			}
		}
	}

	int label_s = label_anchor[0];
	int pos = 1;
	vector<vector<int> > idx_classes(0);
	vector<int> idx_class(1, 0);
	for (int i = pos; i < length; ++i) {
		if (label_anchor[i] != label_s) {
			label_s = label_anchor[i];
			idx_classes.push_back(idx_class);
			idx_class.clear();
			idx_class.push_back(pos_gt[i]);
		}
		else {
			idx_class.push_back(pos_gt[i]);
		}
	}
	idx_classes.push_back(idx_class);

	Manc = cv::Mat(length, idx_classes.size(), CV_32FC1, cvScalarAll(0));
	float ent_anchor = 0;
	for (int i = 0; i < idx_classes.size(); ++i) {
		ent_anchor += idx_classes[i].size() * log(float(idx_classes[i].size()) / float(length));
		for (int j = 0; j < idx_classes[i].size(); ++j) {
			Manc.at<float>(idx_classes[i][j], i) = 1;
		}
	}

	// step-2: compute entropy for label_predicted
	int* pos_pred = new int[length];
	for (int i = 0; i < length; ++i) {
		pos_pred[i] = i;
	}

	for (int i = 0; i < length; ++i) {
		for (int j = i + 1; j < length; ++j) {
			if (label_predict[i] > label_predict[j]) {
				swap(label_predict[i], label_predict[j]);
				swap(pos_pred[i], pos_pred[j]);
			}
		}
	}

	label_s = label_predict[0];
	pos = 1;
	vector<vector<int> > idxp_classes(0);
	vector<int> idxp_class(1, 0);
	for (int i = pos; i < length; ++i) {
		if (label_predict[i] != label_s) {
			label_s = label_predict[i];
			idxp_classes.push_back(idxp_class);
			idxp_class.clear();
			idxp_class.push_back(pos_pred[i]);
		}
		else {
			idxp_class.push_back(pos_pred[i]);
		}
	}
	idxp_classes.push_back(idxp_class);

	Mpred = cv::Mat(length, idxp_classes.size(), CV_32FC1, cvScalarAll(0));
	float ent_predict = 0;
	for (int i = 0; i < idxp_classes.size(); ++i) {
		ent_predict += idxp_classes[i].size() * log(float(idxp_classes[i].size()) / float(length));
		for (int j = 0; j < idxp_classes[i].size(); ++j) {
			Mpred.at<float>(idxp_classes[i][j], i) = 1;
		}
	}

	// step-3: compute mutual entropy between label_anchor and label_predicted
	//float ment = 0;
	//for (int i = 0; i < idx_classes.size(); ++i) {
	//	int n_i = idx_classes[i].size();
	//	for (int j = 0; j < idxp_classes.size(); ++j) {
	//		int n_j = idxp_classes[j].size();
	//		// count n_h,l
	//		int n_h_l = 0;
	//		for (int m = 0; m < idx_classes[i].size(); ++m) {
	//			for (int n = 0; n < idxp_classes[j].size(); ++n) {
	//				if (idx_classes[i][m] == idxp_classes[j][n])
	//					++n_h_l;
	//			}
	//		}

	//		// calculation
	//		if (n_h_l == 0)
	//			continue;

	//		ment += n_h_l * log(float(length * n_h_l) / float(n_i * n_j));
	//	}
	//}

	// fast algorithm
	// Manc' * Mpred
	cv::Mat Mproduct = Manc.t() * Mpred;
	// compute cross entropy
	float ent_cross = 0;
	for (int i = 0; i < Mproduct.rows; ++i) {
		for (int j = 0; j < Mproduct.cols; ++j) {
			float val = Mproduct.at<float>(i, j) / length;
			if (val > 0)
				ent_cross += Mproduct.at<float>(i, j) * log(val);
		}
	}
	// compute mutual information
	float ment = - (ent_anchor + ent_predict) + ent_cross;

	delete[] pos_gt;
	delete[] pos_pred;
	delete[] label_anchor;
	delete[] label_predict;
	return ment / sqrt(ent_anchor * ent_predict);

}