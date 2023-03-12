#模型訓練等待序列
import library as lib

training_limit=5
while True:
    while lib.db_operate.get_training_count()<training_limit:
        if len(lib.db_operate.get_train_queue_content())!=0:
            model_id=lib.db_operate.train_queue_pop()
            labels=lib.db_operate.label_list(model_id)
            if __name__=='__main__':
                print('start training')
                lib.Thread(target=lib.train_start, args=(model_id, labels)).start()
                
        else:
            break
    lib.time.sleep(10)