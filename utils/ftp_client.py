from ftplib import FTP

def upload_file(file_path, remote_path, server, user, password):
    """上传文件到FTP服务器"""
    ftp = FTP(server)
    ftp.login(user=user, passwd=password)
    
    with open(file_path, 'rb') as file:
        ftp.storbinary(f'STOR {remote_path}', file)
    
    ftp.quit()
    return True