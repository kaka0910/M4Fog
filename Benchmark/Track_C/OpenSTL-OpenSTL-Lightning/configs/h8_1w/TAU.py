method = 'TAU'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'tau'
hid_S = 64
hid_T = 256
N_T = 8
N_S = 4
alpha = 0.1
# training
lr = 1e-4
batch_size = 32
drop_path = 0.1
# sched = 'cosine'
warmup_epoch = 5
epoch = 100