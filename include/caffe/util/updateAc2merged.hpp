float computeAc2merged(float* W_sub_mc, float* W_sub_cm, const int rows, const int cols);
void batchDistance(float* feat, float* d_samples, int* nIdx, const int Knn, const int n, int d);
void entityDistance(float* feat, float* d_samples, const int n, int d);