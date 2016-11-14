if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

net_model = 'D:\caffe-windows-master\caffe-windows-master\models\vt_autoencoder\deploy.prototxt';
net_weights = 'D:\caffe-windows-master\caffe-windows-master\models\vt_autoencoder\autoencoder_iter_30000.caffemodel';
phase = 'test'; % run with phase test (so that dropout isn't applied)
% Initialize a network
train_net = caffe.Net(net_model, net_weights, phase);

%%%%%%%%%extract the train features
output_blob_index = train_net.name2blob_index('encode4');%feature blob
output_blob = train_net.blob_vec(output_blob_index);
output_label_index = train_net.name2blob_index('label');
output_label = train_net.blob_vec(output_label_index);
feature_train = [];
label_train = [];
num = 50;%data_num / batch_size
for i = 1 : num
    disp(i);
    train_net.forward_prefilled();
    output = output_blob.get_data();
    output = reshape(output,size(output,1)*size(output,2)*size(output,3),size(output,4));
    if isempty(feature_train)
        feature_train = zeros(size(output,1),size(output,2)*num);
        feature_train(:,1:size(output,2)) = output;
        label_train = zeros(1,size(output,2)*num);
        label_train(1:size(output,2)) = output_label.get_data();
    else
        feature_train(:,(i-1)*size(output,2)+1:i*size(output,2)) = output;
        label_train((i-1)*size(output,2)+1:i*size(output,2)) = output_label.get_data();
    end;
end;
%check if has traversed all the data
train_net.forward_prefilled();
output = output_blob.get_data();
output = reshape(output,size(output,1)*size(output,2)*size(output,3),size(output,4));
assert(sum(sum(abs(output - feature_train(:,1:size(output,2)))))==0);