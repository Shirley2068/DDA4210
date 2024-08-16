import glob,random

root = '../data/'

# ori + DF
train_lines = []
test_lines = []
val_lines = []

# 获取原始视频和Deepfakes处理后的视频的图片路径
paths1 = glob.glob(root + 'original_sequences/youtube/c23/faces/*/')
paths1_train = paths1[:370]
paths1_test = paths1[370:440]
paths1_val = paths1[440:]

for folder_path in paths1_train:
    image_files = glob.glob(folder_path + '*.png')  # 获取当前子文件夹下的所有png图片路径
    for image_file in image_files:
        line = image_file + ' 1\n'
        train_lines.append(line)

for folder_path in paths1_test:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        test_lines.append(line)

for folder_path in paths1_val:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        val_lines.append(line)

paths2 = glob.glob(root + 'manipulated_sequences/Deepfakes/c23/faces/*/')
paths2_train = paths2[:370]
paths2_test = paths2[370:440]
paths2_val = paths2[440:]

for folder_path in paths2_train:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        train_lines.append(line)

for folder_path in paths2_test:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        test_lines.append(line)

for folder_path in paths2_val:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        val_lines.append(line)

# 打乱 train_lines 和 test_lines 列表顺序
random.shuffle(train_lines)
random.shuffle(test_lines)
random.shuffle(val_lines)

# 写入到文件 train_DF.txt 和 test_DF.txt
with open('1train_DF.txt', 'w') as f_train, open('1test_DF.txt', 'w') as f_test:
    for line in train_lines:
        f_train.write(line)
    for line in test_lines:
        f_test.write(line)

with open('1val_DF.txt', 'w') as f_val:
    for line in val_lines:
        f_val.write(line)




# ori + F2F
train_lines = []
test_lines = []
val_lines = []

paths1 = glob.glob(root + 'original_sequences/youtube/c23/faces/*/')
paths1_train = paths1[:370]
paths1_test = paths1[370:440]
paths1_val = paths1[440:]

for folder_path in paths1_train:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        train_lines.append(line)

for folder_path in paths1_test:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        test_lines.append(line)

for folder_path in paths1_val:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        val_lines.append(line)

paths2 = glob.glob(root + 'manipulated_sequences/Face2Face/c23/faces/*/')
paths2_train = paths2[:370]
paths2_test = paths2[370:440]
paths2_val = paths2[440:]

for folder_path in paths2_train:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        train_lines.append(line)

for folder_path in paths2_test:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        test_lines.append(line)

for folder_path in paths2_val:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        val_lines.append(line)

random.shuffle(train_lines)
random.shuffle(test_lines)
random.shuffle(val_lines)

with open('1train_F2F.txt', 'w') as f_train, open('1test_F2F.txt', 'w') as f_test:
    for line in train_lines:
        f_train.write(line)
    for line in test_lines:
        f_test.write(line)

with open('1val_F2F.txt', 'w') as f_val:
    for line in val_lines:
        f_val.write(line)



# ori + FS
train_lines = []
test_lines = []
val_lines = []

paths1 = glob.glob(root + 'original_sequences/youtube/c23/faces/*/')
paths1_train = paths1[:370]
paths1_test = paths1[370:440]
paths1_val = paths1[440:]

for folder_path in paths1_train:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        train_lines.append(line)

for folder_path in paths1_test:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        test_lines.append(line)

for folder_path in paths1_val:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        val_lines.append(line)

paths2 = glob.glob(root + 'manipulated_sequences/FaceSwap/c23/faces/*/')
paths2_train = paths2[:370]
paths2_test = paths2[370:440]
paths2_val = paths2[440:]

for folder_path in paths2_train:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        train_lines.append(line)

for folder_path in paths2_test:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        test_lines.append(line)

for folder_path in paths2_val:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        val_lines.append(line)

random.shuffle(train_lines)
random.shuffle(test_lines)
random.shuffle(val_lines)

with open('1train_FS.txt', 'w') as f_train, open('1test_FS.txt', 'w') as f_test:
    for line in train_lines:
        f_train.write(line)
    for line in test_lines:
        f_test.write(line)

with open('1val_FS.txt', 'w') as f_val:
    for line in val_lines:
        f_val.write(line)


# ori + NT
train_lines = []
test_lines = []
val_lines = []

paths1 = glob.glob(root + 'original_sequences/youtube/c23/faces/*/')
paths1_train = paths1[:370]
paths1_test = paths1[370:440]
paths1_val = paths1[440:]

for folder_path in paths1_train:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        train_lines.append(line)

for folder_path in paths1_test:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        test_lines.append(line)

for folder_path in paths1_val:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 1\n'
        val_lines.append(line)

paths2 = glob.glob(root + 'manipulated_sequences/NeuralTextures/c23/faces/*/')
paths2_train = paths2[:370]
paths2_test = paths2[370:440]
paths2_val = paths2[440:]

for folder_path in paths2_train:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        train_lines.append(line)

for folder_path in paths2_test:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        test_lines.append(line)
        
for folder_path in paths2_val:
    image_files = glob.glob(folder_path + '*.png')
    for image_file in image_files:
        line = image_file + ' 0\n'
        val_lines.append(line)

random.shuffle(train_lines)
random.shuffle(test_lines)
random.shuffle(val_lines)

with open('1train_NT.txt', 'w') as f_train, open('1test_NT.txt', 'w') as f_test:
    for line in train_lines:
        f_train.write(line)
    for line in test_lines:
        f_test.write(line)

with open('1val_NT.txt', 'w') as f_val:
    for line in val_lines:
        f_val.write(line)

