import os

data_root = './3sseg_datasplits/voc_splits/'
num_samples = '500'
_files_sup = open(os.path.join(data_root, '_'+num_samples+'_train_supervised_v2.txt')).readlines()
_files_unsup = open(os.path.join(data_root, '_'+num_samples+'_train_unsupervised_v2.txt')).readlines()

def obtaining_filename(im_name):
    im_name = im_name.strip()
    return '/JPEGImages/'+im_name+'.jpg'+' '+'/SegmentationClassAug/'+im_name+'.png\n'

files_sup = open(os.path.join(data_root, ''+num_samples+'_train_supervised_v2.txt'), 'w')
for imname in _files_sup:
    line = obtaining_filename(imname)
    files_sup.write(line)
    
files_sup.close()


files_unsup = open(os.path.join(data_root, ''+num_samples+'_train_unsupervised_v2.txt'), 'w')
for imname in _files_unsup:
    line = obtaining_filename(imname)
    files_unsup.write(line)
    
files_unsup.close()