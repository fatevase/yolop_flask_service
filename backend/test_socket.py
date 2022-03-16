import socket
import time

def check(host,port):
    s = None
    for res in socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM):
        af, socktype, proto, canonname, sa = res
        try:
            s = socket.socket(af, socktype, proto)
        except Exception as e:
            s = None
            continue
        try:
            s.settimeout(2)
            s.connect(sa)
        except Exception as e:
            s.close()
            s = None
            continue
        break
    if s is None:
        return 0
    s.close()
    return 1


def testSocket(checkitem='127.0.0.1:5001'):

    data = dict()
    s_time = time.time()
    data['code'] = 1
    host, port = checkitem.split(':')
    if check(host, port):
        data['dems'] = round(1000*(time.time() - s_time), 2)
    else:
        data['code'] = -1
    return data


if __name__ == '__main__':
    pass

