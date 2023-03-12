import library as lib

predicting_limit=1
while True:
    while lib.db_operate.get_predicting_count()<predicting_limit:
        if len(lib.db_operate.get_predict_queue_content())!=0:
            model_id, image_id=lib.db_operate.predict_queue_pop()
            if __name__=='__main__':
                print('start predicting')
                lib.Thread(target=lib.predict_start, args=(model_id, image_id)).start()
        else:
            break
    lib.time.sleep(1)