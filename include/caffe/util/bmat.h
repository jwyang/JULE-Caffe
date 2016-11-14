#ifndef BMAT_H
#define BMAT_H

#include <iostream>
#include <fstream>
#include <time.h>
#include <string>
#include <stdlib.h>
#include "caffe/common.hpp"

using namespace std;
typedef unsigned char byte;

typedef struct {
	char* value;
	__int32* value_int;
	long long rows;
	long long cols;
	int channels;
	int itemsize;
	long long cursor;
} temp_data;

class bmat
{
public:
	bmat();
	bmat(string filepath);
	bmat(char* pdata);
	void init();
	~bmat();

	bool get_data_type_num(string& dtype,int& dtype_num, int& itemsize);
	bool read_bmat(string& filepath, char* temp);
	bool read_bmat(string& filepath, temp_data& temp, int num_samples);
	bool write_bmat(string& filepath);
	bool write_bmat(string& filepath,unsigned char* data,long long rows,long long cols,string& dtype,int order,bool isTrans);
	bool write_bmat(string& filepath, float* data, long long rows, long long cols, string& dtype, int order, bool isTrans);
	bool print_bmat();
	void clear();
public:
	static const int max_dim=32;
	bool isTrans;
	int dtype_num,itemsize,order;
	long long nbytes,ndim,rows,cols,channels;
	long long shape[max_dim],strides[max_dim];
	
	int _header_length;
	long long* header;
	char* data;
	float* p;
};

#endif