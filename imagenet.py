import boto3
import tarfile
from io import BytesIO
from tqdm import tqdm

# Параметры
bucket_name = "imagenet"
tar_key = "ILSVRC.tar"
s3_prefix = ""  # куда загружать распакованные файлы

# Клиент S3
session = boto3.Session(profile_name="pers")
s3 = session.client('s3', endpoint_url="https://s3-ai.intra.ispras.ru/")

# Получаем объект с S3 как поток
obj = s3.get_object(Bucket=bucket_name, Key=tar_key)
stream = obj['Body']

# Функция для потокового чтения tar без полной загрузки в память
def stream_tar_to_s3(stream, bucket, prefix):
    with tarfile.open(fileobj=stream, mode='r|*') as tar:  # потоковый режим
        for member in tar:
            if member.isfile():
                f = tar.extractfile(member)
                if f:
                    file_data = BytesIO(f.read())
                    key = f"{prefix}{member.name}"
                    s3.upload_fileobj(file_data, bucket, key)
                    print(f"Uploaded {key}")

            #break

# Запуск
stream_tar_to_s3(stream, bucket_name, s3_prefix)