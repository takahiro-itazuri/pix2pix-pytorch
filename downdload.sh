base=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/
files=(cityscapes edges2handbags edges2shoes facades maps)

for item in ${files[@]}; do
  wget ${base}${item}.tar.gz -P data
  tar -xvf data/${item}.tar.gz -C data
  dirs=data/${item}/*
  for dir in ${dirs}; do
    if [ -d ${dir} ]; then
      mkdir ${dir}/data
      mv ${dir}/*jpg ${dir}/data
    fi
  done
done
