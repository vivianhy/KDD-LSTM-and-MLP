# -*- coding: utf-8 -*-

import numpy as np
import csv

def find_index(x,y):
    for a in range(len(y)):
        if(y[a] == x):
            return a
    # return [ a for a in range(len(y)) if y[a] == x]
    
def handleProtocol(input):
    protoclo_list=['tcp','udp','icmp']
    if input[1] in protoclo_list:
        return find_index(input[1],protoclo_list)
        
def handleService(input):
    service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u','echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames','http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap','link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp','ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell','smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i','uucp','uucp_path','vmnet','whois','X11','Z39_50']
    if input[2] in service_list:
        return find_index(input[2],service_list)
        
def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if input[3] in flag_list:
        return find_index(input[3],flag_list)

def handleLabel(input):
    if input[41] == 'normal.':
        return 0
    else:
        return 1
        
def preHandle():
    source_file = "/home/hy/test/a.csv"
    handled_file = "/home/hy/test/b.csv"
    data_to_flie=open(handled_file, 'w')
    with (open(source_file,'r')) as data_from:
        csv_reader = csv.reader(data_from)
        csv_writer = csv.writer(data_to_flie)
        for i in csv_reader:
            temp_line = np.array(i)
            temp_line[1] = handleProtocol(i)         
            temp_line[2] = handleService(i)          
            temp_line[3] = handleFlag(i)             
            temp_line[41] = handleLabel(i)           
            csv_writer.writerow(temp_line)
        data_to_flie.close()
                
if __name__ == '__main__':
    preHandle()
