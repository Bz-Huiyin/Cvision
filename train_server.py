#模型訓練
import library as lib

def train_start(model_id, labels): 
                #data input by [train_queue_waiter.py] 
    lib.db_operate.update_progress(model_id, 'Building', None, None)
    #在database上更新model的狀態（避免訓練時的上傳或者預測）
    lib.hyper_run([i['id'] for i in labels], model_id)
    #create_hyper_py.py里的run
    
    d=lib.local() #threading裡的套件 用於多線程
    d.config = lib.ConfigProto(
        gpu_options = lib.GPUOptions(per_process_gpu_memory_fraction=0.1)
        # device_count = {'GPU': 1}
    )
    d.config.gpu_options.allow_growth = True
    d.graph=lib.Graph()
    d.session=lib.Session(graph=d.graph, config=d.config)
    
    lib.db_operate.update_progress(model_id, 'Training', None, None)
    #在database上更新model的狀態（避免訓練時的上傳或者預測）
    try:
        with d.graph.as_default():
            with d.session.as_default():
                d.model = lib.model_from_json(open('hyper_space/'+model_id+'.json', 'r').read())
                if len(labels)==2:  
                    d.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
                else:
                    d.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                lib.os.remove('hyper_space/'+model_id+'.py')
                lib.os.remove('hyper_space/'+model_id+'.json')
                
                X_train=[]
                X_test=[]
                y_train=[]
                y_test=[]
                for label in labels: #labels從function外部導入
                    x=[]
                    y=[]
                    for img in lib.os.listdir('labels/'+label['id']):  #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
                        x.append('labels/'+label['id']+'/'+img)
                        y.append(labels.index(label))
                    xy=list(zip(x,y))
                    lib.random.shuffle(xy)
                    (x, y)=zip(*xy)
                    X_train_, X_test_, y_train_, y_test_ = lib.train_test_split(x, y, test_size=0.2)
                    X_train+=X_train_
                    X_test+=X_test_
                    y_train+=y_train_
                    y_test+=y_test_
                
                def gen_train(X_train, y_train):
                    while 1:
                        xy=list(zip(X_train,y_train))
                        lib.random.shuffle(xy)
                        (X_train,y_train)=zip(*xy)
                        x=[]
                        y=[]
                        for img_path, clas in zip(X_train, y_train):
                            img=lib.cv2.imread(img_path)
                            img=lib.cv2.resize(img,(100,100),interpolation=lib.cv2.INTER_CUBIC)
                            img=img/255
                            x.append(img)
                            if len(labels)>2:
                                y.append(lib.to_categorical(int(clas), num_classes=len(labels)))
                            else:
                                y.append(int(clas))
                            if len(x)==32:
                                yield (lib.np.array(x), lib.np.array(y))
                                x=[]
                                y=[]
                        if len(x)!=0:
                            yield (lib.np.array(x), lib.np.array(y))
                def gen_val(X_test, y_test):
                    while 1:
                        xy=list(zip(X_test,y_test))
                        lib.random.shuffle(xy)
                        (X_test,y_test)=zip(*xy)
                        x=[]
                        y=[]
                        for img_path, clas in zip(X_test, y_test):
                            img=lib.cv2.imread(img_path)
                            img=lib.cv2.resize(img,(100,100),interpolation=lib.cv2.INTER_CUBIC)
                            img=img/255
                            x.append(img)
                            if len(labels)>2:
                                y.append(lib.to_categorical(int(clas), num_classes=len(labels)))
                            else:
                                y.append(int(clas))
                            if len(x)==32:
                                yield (lib.np.array(x), lib.np.array(y))
                                x=[]
                                y=[]
                        if len(x)!=0:
                            yield (lib.np.array(x), lib.np.array(y))
                
                d.acc=-999
                d.epoch=0
                d.patient=3
                #==========================================================#
                while 1:
                    d.history_train=d.model.fit_generator(gen_train(X_train, y_train), steps_per_epoch=lib.math.ceil(len(X_train)/32), epochs=1, verbose=1)
                    d.history_test=d.model.evaluate_generator(gen_val(X_test, y_test), steps=lib.math.ceil(len(X_test)/32))
                    d.epoch+=1
                    lib.db_operate.update_progress(model_id, str(d.epoch), str(d.history_test[1]), str(d.history_test[0]))
                    if d.history_test[1]>d.acc:
                        d.acc=d.history_test[1]
                        continue
                    else:
                        is_update=False
                        for i in range(d.patient):# train patient times 
                            d.history_train=d.model.fit_generator(gen_train(X_train, y_train), steps_per_epoch=lib.math.ceil(len(X_train)/32), epochs=1, verbose=1)
                            d.history_test=d.model.evaluate_generator(gen_val(X_test, y_test), steps=lib.math.ceil(len(X_test)/32))
                            d.epoch+=1
                            lib.db_operate.update_progress(model_id, str(d.epoch), str(d.history_test[1]), str(d.history_test[0]))
                            if d.history_test[1]>d.acc:
                                is_update=True
                                d.acc=d.history_test[1]
                                break
                        if is_update==False:
                            break
                #==========================================================#
                d.model.save('models/'+model_id+'.h5')
                
        #finish fill data to database
        size=lib.os.path.getsize('models/'+model_id+'.h5')
        size=(size/1024)/1024
        class_label=lib.json.dumps({labels.index(label):label['name'] for label in labels}, ensure_ascii=False).replace('"', '\\"')
        lib.db_operate.fill_model_details(model_id, size, d.history_test[1], d.history_test[0], class_label)
        lib.db_operate.train_model_finish(model_id)
        del d.model
    except:
        lib.db_operate.train_model_stop(model_id)
        try:
            lib.os.remove('hyper_space/'+model_id+'.py')
        except:
            pass
        try:
            lib.os.remove('hyper_space/'+model_id+'.json')
        except:
            pass
        try:
            del d.model
        except:
            pass
    del d.graph
    del d.session
    del d
    lib.db_operate.remove_progress(model_id)
    #mail user finish or terminate

def progress_stream(model):
    while 1:
        try:
            yield lib.json.dumps(lib.db_operate.get_progress(model)[0], ensure_ascii=False)+'\n'
        except:
            yield 'finish\n'
            break
        lib.time.sleep(0.5)


 