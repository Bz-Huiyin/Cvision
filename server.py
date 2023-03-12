#接收網頁端消息和資料並回傳/操作database
import library as lib
from google.oauth2 import id_token
from google.auth.transport import requests
dl_libs=lib.dl_library()  
app = lib.Flask(__name__) #main
lib.CORS(app, resources=r'/*')
lib.os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("done")
def db_result_parse_to_response_string(db_operate_result, message='ok'):  #turn operate result into string
    #isinstance(object, classinfo) check "result"'s datatype
    if isinstance(db_operate_result, list) or isinstance(db_operate_result, dict):
        ok=lib.copy.deepcopy(lib.code_table.ok)
        #code_table.ok={'status':0, 'message':'ok', 'data':''}
        ok['data']=db_operate_result
        #return urllib.parse.quote_plus(json.dumps(ok, ensure_ascii=False))
        return lib.json.dumps(ok, ensure_ascii=False)
    elif db_operate_result=='ok':
        ok=lib.copy.deepcopy(lib.code_table.ok)
        ok['message']=message
        return lib.json.dumps(ok, ensure_ascii=False)
    else:
        return lib.json.dumps(lib.code_table.error_duplicate_name, ensure_ascii=False)

def is_key_exist(key):
    result=lib.db_operate.is_key_exist(key)
    if len(result)==0:
        return False
    else:
        return True

from google.oauth2 import id_token
from google.auth.transport import requests
from flask import request
@app.route('/signin',methods=['POST'])
def signin():
    print("requesting")
    token=request.form['credential']
    #print(token)
    #try:
    idinfo = id_token.verify_oauth2_token(token, requests.Request(), '473107654986-okn62fr0qc2vncukqu8go7ub31uid8kj.apps.googleusercontent.com')
    #print(idinfo)
    #except:
    #    pass
    return ''

@app.route('/register',methods=['POST'])
def register():
    print("getting token")
    idtoken=lib.urllib.parse.unquote(lib.request.form['idtoken'])
    try:
        idinfo=lib.id_token.verify_oauth2_token(idtoken, lib.google_requests.Request(), '473107654986-okn62fr0qc2vncukqu8go7ub31uid8kj.apps.googleusercontent.com')
        result=lib.db_operate.register(idinfo['email'], idinfo['name'])
        return db_result_parse_to_response_string(result)
    except:
        return lib.json.dumps(lib.code_table.error_login_fail, ensure_ascii=False)

@app.route('/get_key',methods=['POST'])  
def get_key():  
    idtoken=lib.urllib.parse.unquote(lib.request.form['idtoken']) #two data from index.php
    clientid=lib.urllib.parse.unquote(lib.request.form['clientid'])
    try:
        print("trying verfiy")
        idinfo=lib.id_token.verify_oauth2_token(idtoken, requests.Request())
        #print("idinfo",idinfo) use for test
        #send data to google for varification
        result=lib.db_operate.get_key(idinfo['email'])
        #return {"key": key}
        if len(result)==0:
            result=lib.db_operate.register(idinfo['email'], idinfo['name'])
            return db_result_parse_to_response_string(result[0])
            #return lib.json.dumps(lib.code_table.error_unregistered, ensure_ascii=False)
        else:
            #print("db_result_parse_to_response_string(result[0]):",db_result_parse_to_response_string(result[0]))
            return db_result_parse_to_response_string(result[0])
            #return type {status,message,data{key}}
    except:
            print("verify error")
            idinfo=lib.id_token.verify_oauth2_token(idtoken, requests.Request())
            return lib.json.dumps(lib.code_table.error_login_fail, ensure_ascii=False)

@app.route('/create_model',methods=['POST'])
def create_model():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_name=lib.urllib.parse.unquote(lib.request.form['model_name'])
    return db_result_parse_to_response_string(lib.db_operate.create_model(key, model_name))

@app.route('/model_list',methods=['POST'])
def model_list():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    id_name_acc_loss=lib.db_operate.model_list(key)
    return db_result_parse_to_response_string(id_name_acc_loss)

@app.route('/get_model_info',methods=['POST'])
def get_model_info():
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if len(lib.db_operate.is_model_exist(model_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    name_size_acc_loss_class_label_share_email=lib.db_operate.get_model_info(model_id)
    print("information:",name_size_acc_loss_class_label_share_email)
    return db_result_parse_to_response_string(name_size_acc_loss_class_label_share_email)

@app.route('/get_label_info',methods=['POST'])
def get_label_info():
    label_id=lib.urllib.parse.unquote(lib.request.form['label_id'])
    if len(lib.db_operate.is_label_exist(label_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    share=lib.db_operate.get_label_info(label_id)   
    return db_result_parse_to_response_string(share)

@app.route('/delete_model',methods=['POST'])
def delete_model():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if len(lib.db_operate.is_model_exist(model_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if key!=lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    result=lib.db_operate.delete_model(model_id)
    return db_result_parse_to_response_string(result)

@app.route('/create_label',methods=['POST'])                                    #! would influence while tarining
def create_label():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if key!=lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    current_model_status=lib.db_operate.get_model_status(model_id)
    if current_model_status==lib.code_table.model_status_first_train or current_model_status==lib.code_table.model_status_improve:
        return lib.json.dumps(lib.code_table.error_model_is_training, ensure_ascii=False)
    label_name=lib.urllib.parse.unquote(lib.request.form['label_name'])
    result=lib.db_operate.create_label(label_name, model_id)
    return db_result_parse_to_response_string(result)

@app.route('/label_list',methods=['POST'])
def label_list():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if key!=lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    labelid_name=lib.db_operate.label_list(model_id)
    #return dict:{id:"(id)",name:"(model name)"}
    return db_result_parse_to_response_string(labelid_name)

@app.route('/delete_label',methods=['POST'])                                    #! would influence while tarining
def delete_label():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    label_id=lib.urllib.parse.unquote(lib.request.form['label_id'])
    if len(lib.db_operate.is_label_exist(label_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if key!=lib.db_operate.get_label_key(label_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    current_model_status=lib.db_operate.get_model_status(lib.db_operate.get_label_model_id(label_id))
    if current_model_status==lib.code_table.model_status_first_train or current_model_status==lib.code_table.model_status_improve:
        return lib.json.dumps(lib.code_table.error_model_is_training, ensure_ascii=False)
    result=lib.db_operate.delete_label(label_id)
    return db_result_parse_to_response_string(result)
#加入圖片
@app.route('/write_image',methods=['POST'])                                    #! would influence while tarining
def write_image(): #處理上傳的圖片(from labelimage.php)
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    label_id=lib.urllib.parse.unquote(lib.request.form['label_id'])
    if is_key_exist(key)==False:
        #key不存在
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    if len(lib.db_operate.is_label_exist(label_id))==0:
        #label不存在
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if key!=lib.db_operate.get_label_key(label_id):
        #key不匹配
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    current_model_status=lib.db_operate.get_model_status(lib.db_operate.get_label_model_id(label_id))
        #請求模型狀態
    if current_model_status==lib.code_table.model_status_first_train or current_model_status==lib.code_table.model_status_improve:
        #模型在訓練中
        return lib.json.dumps(lib.code_table.error_model_is_training, ensure_ascii=False)
    image=lib.urllib.parse.unquote(lib.request.form['base64_image'])
        #ex for unquote :unquote('/El%20Ni%C3%B1o/') yield '/El Niño/'
    try:
        image=lib.base64.b64decode(image)
        #沒問題則嘗試decode
    except:
        return lib.json.dumps(lib.code_table.error_invalid_base64, ensure_ascii=False)
    image = lib.cv2.imdecode(lib.np.frombuffer(image, lib.np.uint8), lib.cv2.IMREAD_COLOR)
    if isinstance(image, lib.np.ndarray) is not True and image is None:
        return lib.json.dumps(lib.code_table.error_not_image_data, ensure_ascii=False)
    # should always write jpg
    image_id=lib.db_operate.id_hash_string()
    lib.cv2.imwrite('labels/'+label_id+'/'+image_id+'.jpg', image)
    return lib.json.dumps(lib.code_table.ok, ensure_ascii=False)

@app.route('/label_images',methods=['POST'])
def label_images():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    label_id=lib.urllib.parse.unquote(lib.request.form['label_id'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    if len(lib.db_operate.is_label_exist(label_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if key!=lib.db_operate.get_label_key(label_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    result=[]
    for img in lib.os.listdir('labels/'+label_id+'/'):
        result.append(lib.re.split('[.]', img)[0])
    return_data=lib.copy.deepcopy(lib.code_table.ok)
    return_data['data']=result
    return lib.json.dumps(return_data, ensure_ascii=False)

@app.route('/get_image',methods=['POST','GET'])
def get_image():#non format return
    key=lib.urllib.parse.unquote(lib.request.args.get('key'))
    label_id=lib.urllib.parse.unquote(lib.request.args.get('label_id'))
    image_id=lib.urllib.parse.unquote(lib.request.args.get('image_id'))
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    if key!=lib.db_operate.get_label_key(label_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    if lib.request.args.get('type') is not None:
        type=lib.urllib.parse.unquote(lib.request.args.get('type'))
    img=lib.cv2.imread('labels/'+label_id+'/'+image_id+'.jpg')
    if lib.request.args.get('type') is not None and type!='full':
        width=300
        size=(width,int(img.shape[0]*(width/img.shape[1])))
        img=lib.cv2.resize(img, size, interpolation=lib.cv2.INTER_AREA)
    img=lib.cv2.imencode('.jpg', img)[1].tostring()
    img=lib.io.BytesIO(img)
    return lib.send_file(img, mimetype='image/jpg')

@app.route('/delete_image',methods=['POST'])                                    #! would influence while tarining
def delete_image():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    label_id=lib.urllib.parse.unquote(lib.request.form['label_id'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    if key!=lib.db_operate.get_label_key(label_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    model_id=lib.db_operate.get_label_model_id(label_id)
    current_model_status=lib.db_operate.get_model_status(model_id)
    if current_model_status==lib.code_table.model_status_first_train or current_model_status==lib.code_table.model_status_improve:
        return lib.json.dumps(lib.code_table.error_model_is_training, ensure_ascii=False)
    image_id=lib.urllib.parse.unquote(lib.request.form['image_id'])
    try:
        lib.os.remove('labels/'+label_id+'/'+image_id+'.jpg')
    except:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    return lib.json.dumps(lib.code_table.ok, ensure_ascii=False)
    
@app.route('/progress_list',methods=['POST'])
def progress_list():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    id_name=lib.db_operate.progress_list(key)
    return db_result_parse_to_response_string(id_name)
    
@app.route('/predictable_model_list',methods=['POST'])
def predictable_model_list():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    id_name_email=lib.db_operate.predictable_model_list(key)
    return db_result_parse_to_response_string(id_name_email)
    
    
'''
from ccc import run as ccc_run
@app.route('/test',methods=['GET'])
def test():
    if __name__ == '__main__':
        #lib.Process(target=hyper_run_test).start()
        lib.Process(target=ccc_run).start()
    return ''
'''

lib.subp.Popen('cmd /c '+'python train_queue_waiter.py', shell=True)
@app.route('/train',methods=['POST'])                                    #! would influence while tarining
def train():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if key!=lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    current_model_status=lib.db_operate.get_model_status(model_id)
    if current_model_status==lib.code_table.model_status_first_train or current_model_status==lib.code_table.model_status_improve:
        return lib.json.dumps(lib.code_table.error_model_is_training, ensure_ascii=False)
    if current_model_status==lib.code_table.model_status_copying:
        return lib.json.dumps(lib.code_table.error_model_is_copying, ensure_ascii=False)
    
    img_num_check=True
    labels=lib.db_operate.label_list(model_id)
    if len(labels)<2:
        return lib.json.dumps(lib.code_table.error_label_count_less_than_two, ensure_ascii=False)
    for label in labels:
        if len(lib.os.listdir('labels/'+label['id']))<10:
            img_num_check=False
            break
    if img_num_check==False:
        return lib.json.dumps(lib.code_table.error_image_less, ensure_ascii=False)
    
    lib.db_operate.train_model(model_id)
    lib.db_operate.update_progress(model_id, 'In waiting queue', None, None)
    
    lib.db_operate.train_queue_add(model_id)
    return lib.json.dumps(lib.code_table.ok, ensure_ascii=False)

@app.route('/progress',methods=['POST'])                                    #! would fail if not training
def progress():#non format return
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if key!=lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    current_model_status=lib.db_operate.get_model_status(model_id)
    if current_model_status is not lib.code_table.model_status_first_train and current_model_status is not lib.code_table.model_status_improve:
        return lib.json.dumps(lib.code_table.error_model_is_not_training, ensure_ascii=False)
    return lib.Response(lib.stream_with_context(lib.progress_stream(model_id)))

@app.route('/terminate_train',methods=['POST'])
def terminate_train():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if key!=lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    current_model_status=lib.db_operate.get_model_status(model_id)
    if current_model_status is not lib.code_table.model_status_first_train and current_model_status is not lib.code_table.model_status_improve:
        return lib.json.dumps(lib.code_table.error_model_is_not_training, ensure_ascii=False)
    
    pid=lib.db_operate.get_progress_pid(model_id)
    parent = lib.psutil.Process(pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()
            
    return lib.json.dumps(lib.code_table.ok, ensure_ascii=False)

lib.subp.Popen('cmd /c '+'python predict_queue_waiter.py', shell=True)
@app.route('/predict',methods=['POST'])                                    #! would fail if model untrained
def predict():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if key!=lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    current_model_status=lib.db_operate.get_model_status(model_id)
    if current_model_status==lib.code_table.model_status_copying:
        return lib.json.dumps(lib.code_table.error_model_is_copying, ensure_ascii=False)
    if current_model_status is not lib.code_table.model_status_predictable:
        return lib.json.dumps(lib.code_table.error_untrained_model, ensure_ascii=False)
    image=lib.urllib.parse.unquote(lib.request.form['base64_image'])
    try:
        image=lib.base64.b64decode(image)
    except:
        return lib.json.dumps(lib.code_table.error_invalid_base64, ensure_ascii=False)
    image = lib.cv2.imdecode(lib.np.frombuffer(image, lib.np.uint8), lib.cv2.IMREAD_COLOR)
    if isinstance(image, lib.np.ndarray) is not True and image is None:
        return lib.json.dumps(lib.code_table.error_not_image_data, ensure_ascii=False)
    image_id=lib.db_operate.id_hash_string()
    open('predict_queue/'+image_id+'.base64', 'w').write(lib.urllib.parse.unquote(lib.request.form['base64_image']))
    if __name__ == '__main__':
        #P=lib.Process(target=lib.predict_start, args=(model_id, image))
        #P.start()
        #P.join()
        lib.db_operate.predict_queue_add(model_id, image_id)
    while True:
        try:
            result=lib.db_operate.get_predict(model_id)[0]['result']
            break
        except:
            lib.time.sleep(1)
            continue
    lib.db_operate.remove_predict(model_id)
    return_data=lib.copy.deepcopy(lib.code_table.ok)
    return_data['data']=result
    return lib.json.dumps(return_data, ensure_ascii=False)


#========= TRADE SYSTEM ==========#
def is_model_buyable(model_id):
    result=lib.db_operate.is_model_buyable(model_id)
    if len(result)==0:
        return False
    else:
        return True
@app.route('/share_model',methods=['POST'])
def share_model():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if len(lib.db_operate.is_model_exist(model_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if key!=lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    share=lib.urllib.parse.unquote(lib.request.form['share'])
    result=lib.db_operate.share_model(model_id, share)
    return db_result_parse_to_response_string(result)
@app.route('/model_store',methods=['GET', 'POST'])
def model_store():
    keyword=lib.urllib.parse.unquote(lib.request.form['keyword'])
    result=lib.db_operate.model_store(keyword)
    return db_result_parse_to_response_string(result)
@app.route('/model_samples',methods=['GET', 'POST'])
def model_samples():
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if len(lib.db_operate.is_model_exist(model_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if lib.request.args.get('bound') is not None:
        bound=lib.urllib.parse.unquote(lib.request.args.get('bound'))
        result=lib.db_operate.model_samples(model_id, int(bound))
    else:
        result=lib.db_operate.model_samples(model_id)
    return_data=lib.copy.deepcopy(lib.code_table.ok)
    return_data['data']=result
    return lib.json.dumps(return_data, ensure_ascii=False)
@app.route('/model_import',methods=['POST'])
def model_import():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    model_id=lib.urllib.parse.unquote(lib.request.form['model_id'])
    if len(lib.db_operate.is_model_exist(model_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if is_model_buyable(model_id)==False or key==lib.db_operate.get_model_key(model_id):
        return lib.json.dumps(lib.code_table.error_model_unbuyable, ensure_ascii=False)
    if __name__ == '__main__':
        lib.Process(target=lib.db_operate.model_import, args=(model_id, key)).run()
    return lib.json.dumps(lib.code_table.ok, ensure_ascii=False)
@app.route('/share_label',methods=['POST'])
def share_label():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    label_id=lib.urllib.parse.unquote(lib.request.form['label_id'])
    if len(lib.db_operate.is_label_exist(label_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if key!=lib.db_operate.get_label_key(label_id):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    share=lib.urllib.parse.unquote(lib.request.form['share'])
    result=lib.db_operate.share_label(label_id, share)
    return db_result_parse_to_response_string(result)
@app.route('/label_store',methods=['GET', 'POST'])
def label_store():
    keyword=lib.urllib.parse.unquote(lib.request.form['keyword'])
    result=lib.db_operate.label_store(keyword)
    return db_result_parse_to_response_string(result)
@app.route('/label_samples',methods=['GET', 'POST'])
def label_samples():
    label_id=lib.urllib.parse.unquote(lib.request.form['label_id'])
    if len(lib.db_operate.is_label_exist(label_id))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if lib.request.args.get('bound') is not None:
        bound=lib.urllib.parse.unquote(lib.request.args.get('bound'))
        result=lib.db_operate.label_samples(label_id, int(bound))
    else:
        result=lib.db_operate.label_samples(label_id)
    return_data=lib.copy.deepcopy(lib.code_table.ok)
    return_data['data']=result
    return lib.json.dumps(return_data, ensure_ascii=False)
@app.route('/label_import',methods=['POST'])
def label_import():
    key=lib.urllib.parse.unquote(lib.request.form['key'])
    if is_key_exist(key)==False:
        return lib.json.dumps(lib.code_table.error_key_is_not_exist, ensure_ascii=False)
    from_label=lib.urllib.parse.unquote(lib.request.form['from_label'])
    to_label=lib.urllib.parse.unquote(lib.request.form['to_label'])
    if len(lib.db_operate.is_label_exist(to_label))==0 or len(lib.db_operate.is_label_exist(from_label))==0:
        return lib.json.dumps(lib.code_table.error_object_not_exist, ensure_ascii=False)
    if from_label==to_label:
        return lib.json.dumps(lib.code_table.error_model_unimportable, ensure_ascii=False)
    if key!=lib.db_operate.get_label_key(to_label):
        return lib.json.dumps(lib.code_table.error_invalid_key, ensure_ascii=False)
    if __name__ == '__main__':
        lib.Process(target=lib.db_operate.label_import, args=(from_label, to_label)).run()
    return lib.json.dumps(lib.code_table.ok, ensure_ascii=False)





@app.errorhandler(404)
def page_not_found(e):
    return lib.json.dumps(lib.code_table.error_unknow_service, ensure_ascii=False)

if __name__ == '__main__':
    lib.os.system('color')
    print('\033[93m'+'\t\t\t\t\tShould Turn On SQL Server!!'+'\033[0m')
    print('\033[93m'+'\t\t\t\t\tShould Turn On SQL Server!!'+'\033[0m')
    print('\033[93m'+'\t\t\t\t\tShould Turn On SQL Server!!'+'\033[0m')
    print('\033[93m'+'\t\t\t\t\tShould Turn On SQL Server!!'+'\033[0m')
    print('\033[93m'+'\t\t\t\t\tShould Turn On SQL Server!!'+'\033[0m')
    print()
    print('\033[93m'+'\t\t\t\t\tAnd Adjust System Time correctly!!'+'\033[0m')

    app.run(host='0.0.0.0',port=8080)  #START SERVER
