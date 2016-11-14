#include "caffe/util/bmat.h"

bmat::bmat()
{
	init();
}

void bmat::init()
{
	itemsize = 0;
	nbytes = 0;
	ndim = 0;
	rows = 0;
	cols = 0;
	channels = 0;
	header = NULL;
	data = NULL;

	for (int i=0;i<max_dim;i++)
	{
		shape[i] = 0;
		strides[i] = 0;
	}

	

}
bmat::bmat(string filepath)
{
	init();
	read_bmat(filepath, data);
	print_bmat();
}

bmat::bmat(char* pdata)
{

}

bmat::~bmat()
{
	if(header)
		delete [] header;
	if(data)
		delete [] data;
}

bool bmat::get_data_type_num(string& dtype,int& dtype_num, int& itemsize)
{
	if(dtype=="int8" || dtype=="char"){
		dtype_num = 0;
		itemsize = 1;
	}
	else if(dtype=="uint8" || dtype=="unsigned char" || dtype=="uchar"){
		dtype_num = 1;
		itemsize = 1;
	}
	else if(dtype=="int16" || dtype=="short"){
		dtype_num = 2;
		itemsize = 2;
	}
	else if(dtype=="uint16" || dtype=="unsigned short"){
		dtype_num = 3;
		itemsize = 2;
	}
	else if(dtype=="int32"|| dtype=="int" || dtype=="long"){
		dtype_num = 4;
		itemsize = 4;
	}
	else if(dtype=="uint32"|| dtype=="unsingend int" || dtype=="unsigned long"){
		dtype_num = 5;
		itemsize = 4;
	}
	else if(dtype=="int64" || dtype=="long long"){
		dtype_num = 6;
		itemsize = 8;
	}
	else if(dtype=="uint64"|| dtype=="unsingend long long"){
		dtype_num = 7;
		itemsize = 8;
	}
	else if(dtype=="float" || dtype=="float32" || dtype=="single"){
		dtype_num = 8;
		itemsize = 4;
	}
	else if(dtype=="double" || dtype=="float64" ){
		dtype_num = 9;
		itemsize = 8;
	}
	else{
		cout<<"data type error!"<<endl;
		return false;
	}
	return true;
}

bool bmat::read_bmat(string& filepath, char* temp)
{
	// cout<<"Reading bmat file: "<<filepath<<endl;
	ifstream fin(filepath,ios::binary);
	if (!fin || !fin.is_open()) {
		cout << "Read binary mat failed." << filepath << endl;
		return false;
	}

	//=====================header info============================//
	// header 21x32 matrix uint64
	fin.read((char*)(&_header_length),8);  //read header_length1
    long long* header = new long long[_header_length];
	fin.read((char*)(header+1),sizeof(long long)*(_header_length-1));

	dtype_num = header[1];  //dtype_num  8:float
	order = header[2];
	isTrans = header[3];

	int bias = 16;
	itemsize = header[bias];
	nbytes = header[bias+1];
	ndim = header[bias+2];

	if (isTrans){
		cols = header[bias+3];
		rows = header[bias+4];
	}
	else{
		rows = header[bias+3];
		cols = header[bias+4];
	}
	channels = header[bias+5];
	for(int i=0;i<ndim;i++)
		shape[i] = header[bias+3+i];

	strides[0] = cols;
	strides[1] = strides[0]*rows;
	for(int i=2;i<ndim;i++)
		strides[i] = strides[i-1]*shape[i];

	//=====================header info end============================//
	// temp = new char[nbytes];
	fin.read((char*)temp,nbytes); 
	fin.close();

	//end = clock();
	//printf("read binary file took %0.2f sec\n",(end-start)/1000.0);
	delete [] header;
	return true;
}

bool bmat::read_bmat(string& filepath, temp_data& temp, int num_samples)
{
	// cout << "Reading bmat file: " << filepath << endl;
	ifstream fin(filepath, ios::binary);
	if (!fin || !fin.is_open()) {
		temp.cols = 0;
		temp.rows = 0;
		cout << "Read binary mat failed." << filepath << endl;
		return false;
	}

	//=====================header info============================//
	// header 21x32 matrix uint64
	fin.read((char*)(&_header_length), 8);  //read header_length1
	long long* header = new long long[_header_length];
	fin.read((char*)(header + 1), sizeof(long long)*(_header_length - 1));

	dtype_num = header[1];  //dtype_num  8:float
	order = header[2];
	isTrans = header[3];

	int bias = 16;
	itemsize = header[bias];
	nbytes = header[bias + 1];
	ndim = header[bias + 2];

	rows = header[bias + 3];
	cols = header[bias + 4];
	nbytes = nbytes > cols * rows * itemsize ? cols * rows * itemsize : nbytes;

	channels = header[bias + 5];
	for (int i = 0; i < ndim; i++)
		shape[i] = header[bias + 3 + i];

	strides[0] = cols;
	strides[1] = strides[0] * rows;
	for (int i = 2; i<ndim; i++)
		strides[i] = strides[i - 1] * shape[i];

	//=====================header info end============================//
	temp.rows = rows;
	temp.cols = cols;
	temp.channels = channels;
	temp.itemsize = itemsize;
	temp.value = new char[nbytes];
	temp.value_int = new int[nbytes / 4];
	if (itemsize == 1) {
		fin.read((char*)temp.value, nbytes);
	} else if (itemsize == 4){
		fin.read((char*)temp.value_int, nbytes);
	}

	fin.close();

	//end = clock();
	//printf("read binary file took %0.2f sec\n",(end-start)/1000.0);
	delete[] header;
	return true;
}

bool bmat::write_bmat(string& filepath, float* data, long long rows, long long cols, string& dtype, int order, bool isTrans)
{
	cout << "Writing bmat file: " << filepath << endl;
	//if ((order + isTrans) % 2 != 0)
	//cout << "[ERROR] order+isTrans!=2" << endl;

	int dtype_num, itemsize;
	long long nbytes, ndim, channels;
	int bias = 16;

	// get_data_type_num(dtype, dtype_num, itemsize);

	nbytes = rows*cols*sizeof(float);

	long long header[32];
	int header_length = 32;
	for (int i = 0; i<header_length; i++)
		header[i] = 0;

	header[0] = header_length;
	header[1] = 8;  //dtype_num  8:float
	header[2] = order;
	header[3] = isTrans;

	header[bias] = 4;
	header[bias + 1] = nbytes;
	header[bias + 2] = 2;
	if (isTrans){
		header[bias + 3] = cols;
		header[bias + 4] = rows;
	}
	else{
		header[bias + 3] = rows;
		header[bias + 4] = cols;
	}

	//FILE* fid = fopen(filepath.c_str(),"wb");
	//fwrite(header,1,header_length*8,fid);
	//fwrite(data,1,nbytes,fid);

	//fclose(fid);

	ofstream fout(filepath, ios::binary);
	if (!fout)
	{
		cerr << "open bmat file error!" << endl;
		abort();//退出程序
	}
	fout.write((char*)header, header_length * sizeof(long long));
	fout.write((char*)data, nbytes);
	fout.close();
	return true;
}

bool bmat::write_bmat(string& filepath,unsigned char* data,long long rows,long long cols,string& dtype,int order,bool isTrans)
{
	// cout<<"Writing bmat file: "<<filepath<<endl;
	cout << "write bmat " << filepath << endl;
	// if( (order+isTrans)%2!=0 )
		// cout<<"[ERROR] order+isTrans!=2"<<endl;

	int dtype_num,itemsize;
	long long nbytes,ndim,channels;
	int bias = 16;	

	get_data_type_num(dtype,dtype_num,itemsize);	

	nbytes = rows*cols*itemsize;
    
	long long header[32];
	int header_length = 32;
	for (int i=0;i<header_length;i++)
		header[i] = 0;

	header[0] = 32;
	header[1] = dtype_num;  //dtype_num  8:float
	header[2] = order;
	header[3] = isTrans;

    header[bias] = itemsize;
    header[bias+1] = nbytes;
    header[bias+2] = 2;
	if (isTrans){
		header[bias+3] = cols;
		header[bias+4] = rows;
	}
	else{
		header[bias+3] = rows;
		header[bias+4] = cols;
	}

	//FILE* fid = fopen(filepath.c_str(),"wb");
	//fwrite(header,1,header_length*8,fid);
	//fwrite(data,1,nbytes,fid);

	//fclose(fid);

	ofstream fout(filepath,ios::binary);
	if(!fout)
	{
		cerr<<"open bmat file error!"<<endl;
		abort( );//退出程序
	}
	fout.write((char*)header,header_length*8);
	fout.write((char*)data,nbytes);
	fout.close( );
	return true;
}

void bmat::clear() {
	delete[]data;
}

bool bmat::print_bmat()
{
	float* p = (float*)data;
	for(int i=0;i<rows;i++)
	{
		for (int j=0;j<cols;j++)
			printf("%f ",p[i*strides[0] + j]);
		
		printf("\n");
	}
	return true;
}