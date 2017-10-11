# coding: utf-8
import configparser
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.application import MIMEApplication
import mimetypes
import os

class MyMail:
    def __init__(self, mail_config_file):
        config = configparser.ConfigParser()
        config.read(mail_config_file)
 
        self.smtp = smtplib.SMTP()
        self.login_user = config.get('SMTP', 'login_user')
        self.login_pwd = config.get('SMTP', 'login_pwd')
        self.from_addr = config.get('SMTP', 'from_addr')
        self.to_addrs = config.get('SMTP', 'to_addrs')
        self.host = config.get('SMTP', 'host')
        self.port = config.get('SMTP', 'port')
 
    # 连接到服务器
    def connect(self):
        self.smtp.connect(self.host, self.port)
 
    # 登陆邮件服务器
    def login(self):
        try:
            self.smtp.login(self.login_user, self.login_pwd)
        except Exception as e:
            print('%s' % e)
 
    # 发送邮件
    def send_mail(self, mail_subject, mail_content, attachment_path_set):
         # 构造MIMEMultipart对象做为根容器
        msg = MIMEMultipart()
        msg['From'] = self.from_addr
        # msg['To'] = self.to_addrs        

        msg['To'] = ','.join(eval(self.to_addrs))
        msg['Subject'] = mail_subject
 
        # 添加邮件内容
        content = MIMEText(mail_content, _charset='gbk')
        msg.attach(content)
 
        for attachment_path in attachment_path_set:
            if os.path.isfile(attachment_path): # 如果附件存在
                type, coding = mimetypes.guess_type(attachment_path)
                if type == None:
                    type = 'application/octet-stream'
 
                major_type, minor_type = type.split('/', 1)
                with open(attachment_path, 'rb') as file:
                    if major_type == 'text':
                        attachment = MIMEText(file.read(), _subtype=minor_type)
                    elif major_type == 'image':
                        attachment = MIMEImage(file.read(),  _subtype=minor_type)
                    elif major_type == 'application':
                        attachment = MIMEApplication(file.read(), _subtype=minor_type)
                    elif major_type == 'audio':
                        attachment = MIMEAudio(file.read(), _subtype=minor_type)
 
                # 修改附件名称
                attachment_name = os.path.basename(attachment_path)
                attachment.add_header('Content-Disposition', 'attachment', filename=('gbk', '', attachment_name))
 
                msg.attach(attachment)
 
        # 得到格式化后的完整文本
        full_text = msg.as_string()
 
        # 发送邮件
        self.smtp.sendmail(self.from_addr, eval(self.to_addrs), full_text)
 
    # 退出
    def quit(self):
        self.smtp.quit()


config_path = os.path.dirname(__file__) + '/mail.conf'
mymail = MyMail(config_path)


if __name__ == '__main__':
    mymail = MyMail('./mail.conf')
    mymail.connect()
    mymail.login()
    mail_content = 'hello,亲，这是一封测试邮件，收到请回复^^ 2014'
    mymail.send_mail('邮件标题--亲，收到一份邮件，请及时查收', mail_content, {})
    mymail.quit()