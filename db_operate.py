#資料庫操作
import library as lib
def register(mail, name):
    command='select user.key from user where email='+list_to_value_string([mail])
    result=do(command)
    if len(result)==0:
        key=id_hash_string()
        command='insert into user values('+list_to_value_string([key, mail, name])+')'
        do(command)
        command='select user.key from user where email='+list_to_value_string([mail])
        result=do(command)
        return result
    else:
        return result
    
def get_key(mail):
    command='select user.key from user where email='+list_to_value_string([mail])
    result=do(command)
    return result

def is_key_exist(key):
    command='select * from user where user.key='+list_to_value_string([key])
    result=do(command)
    return result

def create_model(key, name):
    id=id_hash_string()
    command='insert into model(`id`, `key`, `name`) values('+list_to_value_string([id, key, name])+')'
    result=do(command)
    return result

def model_list(key):
    command='select id, name, acc, loss from model where model.key='+list_to_value_string([key])+' and train_status not in (1,2)'
    result=do(command)
    return result

def get_model_info(model_id):
    command='SELECT `model`.`name`, `model`.`size`, `model`.`acc`, `model`.`loss`, `model`.`class_label`, `model`.`share`, `user`.`email` FROM `model`, `user` WHERE `model`.`key` = `user`.`key` AND `model`.`id` = '+list_to_value_string([model_id])
    result=do(command)
    return result

def get_label_info(label_id):
    command='select share from label where id='+list_to_value_string([label_id])
    result=do(command)
    return result

def get_model_key(model_id):
    command='select model.key from model where id='+list_to_value_string([model_id])
    result=do(command)
    if len(result)==0:
        return ''
    return result[0]['key']
        
def delete_model(model_id):
    labelid_name=label_list(model_id)
    command='delete from model where id='+list_to_value_string([model_id])
    result=do(command)
    try:
        lib.os.remove('models/'+model_id+'.h5')
    except:
        pass
    for label in labelid_name:
        lib.shutil.rmtree('labels/'+label['id'])
    return result

def is_model_exist(model_id):
    command='select * from model where model.id='+list_to_value_string([model_id])
    result=do(command)
    return result
    
def rename_model(id, newname):
    pass

def add_model_user(id,user):
    pass
    
def model_size(id):
    pass

def create_label(name, model_id):
    id=id_hash_string()
    command='insert into label values('+list_to_value_string([id, name, model_id, '0'])+')'
    result=do(command)
    lib.os.mkdir('labels/'+id)
    return result

def rename_label(id, newname):
    pass

def label_list(model_id):
    command='select id, name from label where model_id='+list_to_value_string([model_id])
    result=do(command)
    return result

def delete_label(label_id):
    command='delete from label where id='+list_to_value_string([label_id])
    result=do(command)
    lib.shutil.rmtree('labels/'+label_id)
    return result

def is_label_exist(label_id):
    command='select * from label where id='+list_to_value_string([label_id])
    result=do(command)
    return result

def get_label_key(label_id):
    command='select model.key from model, label where label.model_id=model.id and label.id='+list_to_value_string([label_id])
    result=do(command)
    if len(result)==0:
        return ''
    return result[0]['key']

def get_label_model_id(label_id):
    command='select model_id from label where id='+list_to_value_string([label_id])
    result=do(command)
    if len(result)==0:
        return ''
    return result[0]['model_id']

def progress_list(key):
    command='select id, name from model where model.key='+list_to_value_string([key])+' and train_status in (1,2)'
    result=do(command)
    return result

def predictable_model_list(key):
    command='SELECT `model`.`id`, `model`.`name`, `user`.`email` FROM `model`, `user` WHERE `model`.`key` = `user`.`key` AND `model`.`key` = '+list_to_value_string([key])+' AND train_status = '+list_to_value_string([lib.code_table.model_status_predictable])
    result=do(command)
    return result

def train_model(model_id):  #get model info from database
    current_model_status=get_model_status(model_id)
    if current_model_status==lib.code_table.model_status_None:  #havn't trained yet
        command='update model set train_status='+list_to_value_string([lib.code_table.model_status_first_train])+' where id='+list_to_value_string([model_id])
    elif current_model_status==lib.code_table.model_status_predictable:  #already trained
        command='update model set train_status='+list_to_value_string([lib.code_table.model_status_improve])+' where id='+list_to_value_string([model_id])
    result=do(command)
    return result
def train_model_stop(model_id):
    current_model_status=get_model_status(model_id)
    if current_model_status==lib.code_table.model_status_first_train:
        command='update model set train_status='+list_to_value_string([lib.code_table.model_status_None])+' where id='+list_to_value_string([model_id])
    elif current_model_status==lib.code_table.model_status_improve:
        command='update model set train_status='+list_to_value_string([lib.code_table.model_status_predictable])+' where id='+list_to_value_string([model_id])
    result=do(command)
    return result
def train_model_finish(model_id):
    command='update model set train_status='+list_to_value_string([lib.code_table.model_status_predictable])+' where id='+list_to_value_string([model_id])
    result=do(command)
    return result
def get_model_status(model_id):
    command='select train_status from model where id='+list_to_value_string([model_id])
    result=do(command)
    return result[0]['train_status']
def fill_model_details(model_id, size, acc, loss, class_label):
    command='update model set size='+list_to_value_string([size])+', acc='+list_to_value_string([acc])+', loss='+list_to_value_string([loss])+', class_label='+list_to_value_string([class_label])+' where id='+list_to_value_string([model_id])
    result=do(command)
    return result

def get_class_label(model_id):
    command='select class_label from model where id='+list_to_value_string([model_id])
    result=do(command)
    return result[0]['class_label']

#=== lists ===#
def get_progress(model_id):
    command='select epoch, acc, loss from progress_list where model_id='+list_to_value_string([model_id])
    result=do(command)
    return result
def update_progress(model_id, epoch, acc, loss):
    if len(get_progress(model_id))==0:
        if acc==None:
            command='insert into progress_list(model_id, epoch) values('+list_to_value_string([model_id, epoch])+')'
        else:
            command='insert into progress_list(model_id, epoch, acc, loss) values('+list_to_value_string([model_id, epoch, acc, loss])+')'
        result=do(command)
    else:
        if acc==None:
            command='update progress_list set epoch='+list_to_value_string([epoch])+' where model_id='+list_to_value_string([model_id])
        else:
            command='update progress_list set epoch='+list_to_value_string([epoch])+', acc='+list_to_value_string([acc])+', loss='+list_to_value_string([loss])+' where model_id='+list_to_value_string([model_id])
        result=do(command)
    return result
def optim_progress(model_id, pid):
    command='update progress_list set pid='+list_to_value_string([pid])+' where model_id='+list_to_value_string([model_id])
    result=do(command)
    return result
def get_progress_pid(model_id):
    command='select pid from progress_list where model_id='+list_to_value_string([model_id])
    result=do(command)
    return result[0]['pid']
def remove_progress(model_id):
    command='delete from progress_list where model_id='+list_to_value_string([model_id])
    result=do(command)
    return result
def add_predict(model_id, predict):
    command='insert into predict_list values('+list_to_value_string([model_id, predict])+')'
    result=do(command)
    return result
def get_predict(model_id):
    command='select result from predict_list where model_id='+list_to_value_string([model_id])
    result=do(command)
    return result
def remove_predict(model_id):
    command='delete from predict_list where model_id='+list_to_value_string([model_id])
    result=do(command)
    return result
    



#========== TRADE SYSTEM ==========#
def is_model_buyable(model_id):
    command='select * from model where (train_status='+list_to_value_string([lib.code_table.model_status_improve])+' OR train_status='+list_to_value_string([lib.code_table.model_status_predictable])+') and share='+list_to_value_string(['1'])
    result=do(command)
    return result
def share_model(model_id, share):
    command='update model set share='+list_to_value_string([share])+' where id='+list_to_value_string([model_id])
    result= do(command)
    return result
def model_store(keyword):
    command='select id, model.name, email from model, user where `model`.`key`=`user`.`key` and (train_status='+list_to_value_string([lib.code_table.model_status_improve])+' OR train_status='+list_to_value_string([lib.code_table.model_status_predictable])+') and share='+list_to_value_string(['1'])+' and (`model`.`name` LIKE BINARY "%'+keyword+'%" or `user`.`email` LIKE BINARY "%'+keyword+'%")'
    result=do(command)
    for i in result:
        i['description']='None'
    return result
def model_samples(model_id, bound=None):
    samples=[]
    for j in label_list(model_id):
        image_id=lib.os.listdir('labels/'+j['id'])[0]
        if bound==None:
            samples.append({
                    'name': j['name'],
                    'base64': str(lib.base64.b64encode(open('labels/'+j['id']+'/'+image_id, 'rb').read()).decode())
                    })
        else:
            img=lib.cv2.imread('labels/'+j['id']+'/'+image_id)
            if img.shape[0]>img.shape[1]:
                img=lib.cv2.resize(img, (int(bound*img.shape[1]/img.shape[0]), bound), interpolation=lib.cv2.INTER_AREA)
            elif img.shape[0]<img.shape[1]:
                img=lib.cv2.resize(img, (bound, int(bound*img.shape[0]/img.shape[1])), interpolation=lib.cv2.INTER_AREA)
            lib.cv2.imwrite('D:/Desktop/test.jpg', img)
            img=lib.cv2.imencode('.jpg', img)[1].tostring()
            base64=str(lib.base64.b64encode(img).decode())
            samples.append({
                    'name': j['name'],
                    'base64': base64
                    })
    return samples
def model_import(model_id, key):
    new_model_id=id_hash_string()
    lib.copyfile('models/'+model_id+'.h5', 'models/'+new_model_id+'.h5')    
    info=get_model_info(model_id)[0]
    command='insert into model values('+list_to_value_string([new_model_id, key, info['name'], info['size'], info['acc'], info['loss'], str(lib.code_table.model_status_copying), info['class_label'].replace('"', '\\"'), '0'])+')'
    result=do(command)
    old_labels=label_list(model_id)
    for i in old_labels:
        new_label_id=id_hash_string()
        command='insert into label values('+list_to_value_string([new_label_id, i['name'], new_model_id, '0'])+')'
        do(command)
        lib.os.mkdir('labels/'+new_label_id)
        open('labels/'+new_label_id+'/'+id_hash_string()+'.jpg', 'wb').write(open('labels/'+i['id']+'/'+lib.os.listdir('labels/'+i['id'])[0], 'rb').read())
    
    command='update model set train_status='+list_to_value_string(['0'])+' where id='+list_to_value_string([new_model_id])
    result=do(command)
    return result
def share_label(label_id, share):
    command='update label set share='+list_to_value_string([share])+' where id='+list_to_value_string([label_id])
    result= do(command)
    return result
def label_store(keyword):
    command='select label.id, label.name, email from label, model, user where `label`.`model_id`=`model`.`id` and `model`.`key`=`user`.`key` and `label`.`share`='+list_to_value_string(['1'])+' and (`label`.`name` LIKE BINARY "%'+keyword+'%" OR `model`.`name` LIKE BINARY "%'+keyword+'%" OR `user`.`email` LIKE "%'+keyword+'%")'
    result=do(command)
    filter_result=[]
    for i in result:
        image_list=lib.os.listdir('labels/'+i['id'])
        if len(image_list)!=0:
            i['description']=str(len(image_list))+'張'
            filter_result.append(i)
    return filter_result
def label_samples(label_id, bound=None):
    samples=[]
    image_list=lib.os.listdir('labels/'+label_id)
    try:
        for i in range(3):
            if bound==None:
                samples.append({
                        'name': '',
                        'base64': str(lib.base64.b64encode(open('labels/'+label_id+'/'+image_list[i], 'rb').read()).decode())
                        })
            else:
                img=lib.cv2.imread('labels/'+label_id+'/'+image_list[i])
                if img.shape[0]>img.shape[1]:
                    img=lib.cv2.resize(img, (int(bound*img.shape[1]/img.shape[0]), bound), interpolation=lib.cv2.INTER_AREA)
                elif img.shape[0]<img.shape[1]:
                    img=lib.cv2.resize(img, (bound, int(bound*img.shape[0]/img.shape[1])), interpolation=lib.cv2.INTER_AREA)
                img=lib.cv2.imencode('.jpg', img)[1].tostring()
                base64=str(lib.base64.b64encode(img).decode())
                samples.append({
                        'name': '',
                        'base64': base64
                        })
    except:
        pass
    return samples
def label_import(source_label, destination_label):
    for file in lib.os.listdir('labels/'+source_label):
        lib.copyfile('labels/'+source_label+'/'+file, 'labels/'+destination_label+'/'+file)


#======= training queue ========#
def get_training_count():
    command='select model_id from progress_list where epoch!='+list_to_value_string(['In waiting queue'])
    result=do(command)
    return len(result)
def train_queue_add(model_id):
    ms=int(round(lib.time.time() * 1000))
    command='insert into train_queue values('+list_to_value_string([model_id, str(ms)])+')'
    result=do(command)
    return result
def get_train_queue_content():
    command='select model_id from `train_queue` ORDER by `ms` ASC'
    result=do(command)
    return result
def train_queue_pop():
    q=get_train_queue_content()
    first=q[0]['model_id']
    command='delete from train_queue where model_id='+list_to_value_string([first])
    do(command)
    return first
#======== predict queue ========#
def get_predicting_count():
    command='select model_id from progress_list'
    result=do(command)
    return len(result)
def predict_queue_add(model_id, image_id):
    ms=int(round(lib.time.time() * 1000))
    command='insert into predict_queue values('+list_to_value_string([model_id, image_id, str(ms)])+')'
    result=do(command)
    return result
def get_predict_queue_content():
    command='select model_id, image_id from `predict_queue` ORDER by `ms` ASC'
    result=do(command)
    return result
def predict_queue_pop():
    q=get_predict_queue_content()
    first=(q[0]['model_id'], q[0]['image_id'])
    command='delete from predict_queue where model_id='+list_to_value_string([first[0]])
    do(command)
    return first




def id_hash_string():
    ms=int(round(lib.time.time() * 1000))
    id = str(ms)+'-'+str(lib.uuid.uuid4())
    return id.upper()
cnxpool=lib.MySQLConnectionPool(pool_name = "mypool", pool_size = 10,host="localhost",database='cvision',user="root",passwd="")
def do(command):
    db = cnxpool.get_connection()  #connect to mysql
    cursor=db.cursor()
    try:
        cursor.execute(command)
        if cursor.description==None:#not query
            db.commit()
            cursor.close()
            db.close()
            return 'ok'
        columns = [col[0] for col in cursor.description]
        dictionary = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        db.close()
        return dictionary
    except lib.connector.IntegrityError as err:
        #error no reference: https://dev.mysql.com/doc/refman/8.0/en/server-error-reference.html
        cursor.close()
        db.close()
        return str(err.errno)
    cursor.close()
    db.close()
def list_to_value_string(list):
    string=''
    for value in list:
        string+='"'+str(value)+'",'
    string=string[:-1]
    return string

    






