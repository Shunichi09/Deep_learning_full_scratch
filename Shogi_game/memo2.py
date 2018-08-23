# ミニバッチの各サンプルの読み込み開始位置を計算
time_size = 5 # BPTTをいくつで打ち切るのか
batch_size = 10 # バッチサイズ
data_size = 100000 # 時系列データの長さ
max_epoch = # 最大エポック数

max_iters = data_size // (batch_size * time_size) # 現実的に繰り返せる回数（データの数），ランダムに使うわけではない！！，今回は99
jump = (data_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)] # ずらす分

for epoch in range(max_epoch):
    for iter in range(max_iters): # この回数繰り返すとデータをすべて見たことになる
        # ミニバッチの取得
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size): # BPTTを考慮していれる
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size] # 
                batch_t[i, t] = ts[(offset + time_idx) % data_size] # 

        # 勾配を求め、パラメータを更新
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1


batch_size = 1 # 何個データを取り出すか，今回は1にしてる
data_size = len(x) # 行×列の行がでるイメージ
max_iters = data_size // batch_size # 何個データを取り出すか，今回は1にしてる

for epoch in range(max_epoch):
    # シャッフル，# 単純に混ぜただけ
    idx = np.random.permutation(np.arange(data_size))
    xs = x[idx]
    ts = t[idx]

    for iters in range(max_iters):
        # 順番に取り出していく
        batch_x = x[iters]
        batch_t = t[iters]

        sub_data_size = len(batch_x) # 行×列の行がでるイメージ

        # ミニバッチの各サンプルの読み込み開始位置を計算
        time_size = 5 # BPTTをいくつで打ち切るのか
        sub_batch_size = 10 # 各データをみる際のバッチサイズ

        sub_max_iters = sub_data_size // (sub_batch_size * time_size) # 現実的に繰り返せる回数（データの数），ランダムに使うわけではない！！，今回は99
        jump = (sub_data_size - 1) // sub_batch_size
        offsets = [i * jump for i in range(sub_batch_size)] # ずらす分

        for sub_iter in range(sub_max_iters): # この回数繰り返すとデータをすべて見たことになる
            # ミニバッチの取得
            sub_batch_x = np.empty((batch_size, time_size), dtype='i')
            sub_batch_t = np.empty((batch_size, time_size), dtype='i')
            for t in range(time_size): # BPTTを考慮していれる
                for i, offset in enumerate(offsets):
                    sub_batch_x[i, t] = xs[(offset + time_idx) % sub_data_size] # 
                    sub_batch_t[i, t] = ts[(offset + time_idx) % sub_data_size] # 

            # 勾配を求め、パラメータを更新
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1