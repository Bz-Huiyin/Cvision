#數據庫/函式庫引用
from flask import Flask, request, send_file, Response, stream_with_context
from flask_cors import CORS
import cv2, base64, json, io, urllib, os, re, copy, requests, time, math, random, shutil, uuid, psutil
#urllib 用來操作URL
#cv2(Open Source Computer Vision-OpenCV)跨平台電腦視覺庫 用於讀取/影像處理圖片
from shutil import copyfile
import numpy as np                                              
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from google.oauth2 import id_token
from google.auth.transport import requests

from multiprocessing import Process, Manager
import subprocess as subp
from threading import local, Thread
#用於多線程的套件

from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.compat.v1 import Graph, Session, ConfigProto, GPUOptions

from sklearn.model_selection import train_test_split

from mysql.connector.pooling import MySQLConnectionPool

import code_table
from train_server import train_start, progress_stream
from prediction_server import predict_start
from create_hyper_py import run as hyper_run
import db_operate



def dl_library():
    return {
            'model_from_json': model_from_json,
            'load_model': load_model,
            'to_categorical': to_categorical,
            'K': K,
            'tf': tf,
            'Graph': Graph,
            'Session': Session
            }