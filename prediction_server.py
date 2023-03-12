import library as lib


lib.K.clear_session()

'''
graph_bear=Graph()
session_bear=Session(graph=graph_bear)
with graph_bear.as_default():
    with session_bear.as_default():
        model_bear=load_model('models/'+'1557822345285-7CC7D436-469D-4DF6-8B8A-BC2C886287D0'+'.h5')
'''

def predict_start(model_id, image_id):
    image=lib.base64.b64decode(open('predict_queue/'+image_id+'.base64', 'r').read())
    image=lib.cv2.imdecode(lib.np.frombuffer(image, lib.np.uint8), lib.cv2.IMREAD_COLOR)
    
    class_label=lib.json.loads(lib.db_operate.get_class_label(model_id))
    image=lib.cv2.resize(image,(100,100),interpolation=lib.cv2.INTER_CUBIC)
    image=image/255
    image=lib.np.array([image])
    d=lib.local()
    d.result=None

    d.config = lib.ConfigProto(
        gpu_options = lib.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # device_count = {'GPU': 1}
    )
    d.config.gpu_options.allow_growth = True
    d.graph=lib.Graph()
    d.session=lib.Session(graph=d.graph, config=d.config)
    with d.graph.as_default():
        with d.session.as_default():
            d.model=lib.load_model('models/'+model_id+'.h5')
            if len(class_label)==2:
                d.result=d.model.predict_classes(image)[0][0]
            else:
                d.result=d.model.predict_classes(image)[0]
                
    '''
    if model_id!='1557822345285-7CC7D436-469D-4DF6-8B8A-BC2C886287D0':
        d.graph=Graph()
        d.session=Session(graph=d.graph)
        with d.graph.as_default():
            with d.session.as_default():
                d.model=load_model('models/'+model_id+'.h5')
                if len(class_label)==2:
                    d.result=d.model.predict_classes(image)[0][0]
                else:
                    d.result=d.model.predict_classes(image)[0]
    else:
        with graph_bear.as_default():
            with session_bear.as_default():
                d.result=model_bear.predict_classes(image)[0]
    '''
    
    del d.model
    del d.graph
    del d.session
    lib.os.remove('predict_queue/'+image_id+'.base64')
    
    lib.db_operate.add_predict(model_id, class_label[str(d.result)])
    #lib.db_operate.add_predict(model_id, '宗')