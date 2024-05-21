import pymysql
import pymysql
import os
import tensorflow as tf 

from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


user = "root"
pin = "haeju6706@"
host = "localhost"
db_name = "fingerprint_db"


def connect_to_database():
    try:
        # MySQL 데이터베이스 연결 설정
        conn = pymysql.connect(
            host=host,
            user=user,
            password=pin,
            database=db_name
        )
        print("MySQL 데이터베이스에 연결되었습니다.")
        return conn
    except pymysql.Error as e:
        print("MySQL 데이터베이스 연결 오류:", e)
        return None

def add_person_info(conn, name, sex, hand_side):
    try:
        with conn.cursor() as cursor:
            # persons 테이블에 새로운 정보를 추가하는 쿼리 실행
            sql = "INSERT INTO persons (name, sex, hand_side) VALUES (%s, %s, %s)"
            cursor.execute(sql, (name, sex, hand_side))
            conn.commit()
        print("사용자 정보가 추가되었습니다.")
    except pymysql.Error as e:
        print("MySQL 데이터베이스 쿼리 오류:", e)


def extract_label(img_path):
    filename, _ = os.path.splitext(os.path.basename(img_path))
    
    subject_id, etc = filename.split('__')
    gender, lr, finger, _ = etc.split('_')
    
    gender = 'Male' if gender == 'M' else 'Female'
    lr = 'left' if lr =='Left' else 'right'
    
    if finger == 'thumb':
        finger_index = 0
    elif finger == 'index':
        finger_index = 1
    elif finger == 'middle':
        finger_index = 2
    elif finger == 'ring':
        finger_index = 3
    elif finger == 'little':
        finger_index = 4
    
    return int(subject_id), gender, lr, finger_index


if __name__ == "__main__":
    # MySQL 데이터베이스에 연결
    conn = connect_to_database()
    if conn:
        # 지문 파일이 저장된 디렉토리 경로
        fingerprint_dir = "./Real"
        # 디렉토리 내의 파일을 순회
        for img_file in os.listdir(fingerprint_dir):
            # 파일의 경로
            img_path = os.path.join(fingerprint_dir, img_file)
            # 파일로부터 정보 추출
            subject_id, gender, lr,_ = extract_label(img_path)
    
        # persons 테이블에 새로운 정보 추가
            add_person_info(conn, subject_id, gender, lr)
        conn.close()