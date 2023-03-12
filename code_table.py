#引用error code
ok={'status':0, 'message':'ok', 'data':''}
error_unknow_service={'status':-1, 'message':'unknow service'}
error_login_fail={'status':100, 'message':'invalid id token, login fail.'}
error_unregistered={'status':200, 'message':'this email is un-registered, please visite cvision website to register.'}
error_key_is_not_exist={'status':300, 'message':'this key is not exist.'}
error_invalid_key={'status':400, 'message':'invalid key.'}
error_google_down={'status':500, 'message':'cannot reach google authenticate server'}
error_database_down={'status':600, 'message':'database server down, mail admin'}
error_server_down={'status':700, 'message':'server down'}

error_object_not_exist={'status':801, 'message':'invalid operate, object not exist'}
error_duplicate_name={'status':802, 'message':'invalid operate, duplicate name'}
error_name_lenght_large_than_100={'status':803, 'message':'invalid operate, length of name over 100.'}
error_capacity_exceed={'status':804, 'message':'invalid operate, capacity exceed'}
error_not_image_data={'status':805, 'message':'invalid operate, not image data.'}
error_invalid_base64={'status':806, 'message':'invalid operate, invalid base64 string.'}
error_label_count_less_than_two={'status':807, 'message':'invalid operate, label count less than two.'}
error_image_less={'status':808, 'message':'invalid operate, include label which num of image less then 10.'}
error_model_is_training={'status':809, 'message':'invalid operate, cannot change anything while model training.'}
error_model_is_not_training={'status':810, 'message':'invalid operate, model is not training.'}
error_untrained_model={'status':811, 'message':'invalid operate, un-trained model.'}
error_model_is_copying={'status':812, 'message':'invalid operate, model is importing.'}

model_status_None=-1
model_status_predictable=0
model_status_first_train=1
model_status_improve=2
model_status_copying=3



#========== TRADE SYSTEM ==========#
error_model_unbuyable={'status':901, 'message':'model un-buyable.'}
error_model_unimportable={'status':902, 'message':'label un-importable.'}


model_permission_denyall=0
model_permission_owner=31 #11111, means all, five permission
model_permission_predict=1
model_permission_browerimage=2
model_permission_train=4
