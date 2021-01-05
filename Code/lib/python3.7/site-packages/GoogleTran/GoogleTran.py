#coding: utf-8

from bs4 import BeautifulSoup
import urllib.request
import urllib
import threading

resultLock = threading.Lock()
maxTranslateConnections = 10
connectionLock = threading.BoundedSemaphore(maxTranslateConnections)

opener = urllib.request.build_opener()
opener.addheaders=[('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:19.0) Gecko/20100101 Firefox/19.0')]


def googletran(sl,tl,text):
    if  encodeLen(text)<1975:
        return partTran(sl,tl,text)
        
            
    else:
        result={}
        divedText=divText(text)
        number=0
        allThread=[]
        for div in divedText:
            connectionLock.acquire()
            t = threading.Thread(target=partTran, args=(sl,tl,div,number,result))
            t.start()
            allThread.append(t)
            number+=1
        for t in allThread:
            t.join()
        longResult=''
        for key in sorted(result):
            longResult+=result[key]
            longResult+='\n'
        return longResult
    
            

def encodeLen(text):
    tempData={'text':text}
    return len(urllib.parse.urlencode(tempData))


def divText(text):
    #对文本分段
    #先按回车分
    #if 回车<1975  ok
    #else 按,或. 分割 找到以.或.分割的最接近1975长度的字符串
    tempText=text.split('\n')
    finalText=[]
    for temp in tempText:
        if encodeLen(temp)<1975:
            finalText.append(temp)
        else:
            assert '.' in temp or '。' in temp
            allPart=[]
            tempPart=''
            if temp.count('.')>temp.count('。'):
                allPart=temp.split('.')
                for p in allPart:
                    assert encodeLen(p)<1975
                    if encodeLen(tempPart+'.'+p)<1975:
                        tempPart+=p
                        tempPart+='.'
                    else:
                        finalText.append(tempPart)
                        tempPart=p
            else:
                allPart=temp.split('。')
                for p in allPart:
                    assert encodeLen(p)<1975
                    if encodeLen(tempPart+'.'+p)<1975:
                        tempPart+=p
                        tempPart+='。'
                    else:
                        finalText.append(tempPart)
                        tempPart=p
    #以回车分割需去掉最后一个空项
    if '\n' in text:
        return finalText[:-1]
    else:
        return finalText
                
            


def partTran(sl,tl,text,number=0,longResult=0):
    data = {'sl':sl,
            'tl':tl,
            'text':text}
    querystring = urllib.parse.urlencode(data)
    request = urllib.request.Request('http://www.translate.google.com?'+ querystring)
    feedBack=opener.open(request).read().decode('utf-8')
    feedBack=feedBack.replace('<br>','##n##')
    soup = BeautifulSoup(feedBack)
    result=soup.find('span', id="result_box").get_text()
    result=result.replace('##n##','\n')
    if longResult!=0:
        resultLock.acquire()
        longResult.update({number:result})
        resultLock.release()
        connectionLock.release()
    else:
        return result
    

if __name__ == '__main__':
    print(googletran('zh-CN','en','你好'))
