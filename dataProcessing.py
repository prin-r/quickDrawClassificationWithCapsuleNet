import tensorflow as tf
import os
import codecs
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import util
import matplotlib.image as mpimg
import imageio

from PIL import Image
from io import BytesIO
import base64
import cv2

from collections import defaultdict

pathData_ = "D:/src/quickDraw/data/"
path_ = "D:/src/quickDraw/test/"
pathImage_ = "D:/src/quickDraw/animalImg"
pathImageText_ = "D:/src/quickDraw/aimgText"

def parse_line(ndjson_line):
  """Parse an ndjson line and return ink (as np array) and classname."""
  sample = json.loads(ndjson_line)
  class_name = sample["word"]
  inkarray = sample["drawing"]

  maxY = np.max(np.max([y[1] for y in inkarray]))
  for e in inkarray:
      e[1] = maxY - e[1]

  shouldFlipX = random.choice([True, False])

  if shouldFlipX:
      maxX = np.max(np.max([x[0] for x in inkarray]))
      for e in inkarray:
          e[0] = maxX - e[0]

  return inkarray, class_name

def testDraw(data, ext):
    plt.figure(figsize=(1, 1))
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    for e in data:
        plt.plot(e[0], e[1], color='black',linewidth=np.random.uniform(1,2))

    plt.savefig(pathImage_ + ext + '.png', bbox_inches='tight', dpi=59, origin='upper',cmap=plt.cm.gray)
    plt.close()

def fromNdjson2Img():

    classesDict = defaultdict(lambda: 0)
    for file in os.listdir(pathImage_):
        classesDict[file] = 1

    isStart = False
    startLine = 22000
    endLine = 44000

    for file in os.listdir(pathData_):
        if file.endswith(".ndjson"):
            fp = pathData_ + file
            fClass = file.replace(".ndjson","")
            if "snake" == fClass:
                isStart = True

            if isStart and fClass in classesDict:
                with codecs.open(fp, 'r', 'utf-8') as f:
                    i = 0
                    for line in f:
                        img , cName = parse_line(line)
                        i += 1
                        if i > startLine:
                            testDraw(img, '/' + cName + '/' + str(i))
                        if i > endLine:
                            break
                        if i%200 == 0:
                            print (cName + ' ' + str(i))

def imageToArray():
    p = "D:/src/quickDraw/1.png"
    pixes = mpimg.imread(p)
    pixes = pixes[:, :, 0]
    plt.imshow(pixes)
    plt.show()
    l = np.reshape(pixes,[4096]).tolist()
    s = ""
    for e in l:
        s += str(e) + ", "

    f = open("D:/src/quickDraw/" + "1.txt", "w")
    f.write(s[:-2])
    f.close()

def textToImage():
    p = "D:/src/quickDraw/1.txt"
    arrText = ""
    with codecs.open(p, 'r', 'utf-8') as f:
        for line in f:
            arrText = line

    arr = np.fromstring(arrText, dtype=float, sep=',')
    arr = np.reshape(arr,[64,64])
    plt.imshow(arr)
    plt.show()
    print (np.shape(arr))



def readVideo():
    filename = 'D://src/quickDraw/test_painting_1.mp4'
    vid = imageio.get_reader(filename, 'ffmpeg')

    for i, im in enumerate(vid):
        plt.imshow(im[:, :, 0])
        plt.show(block=False)

def testDecodeBase64():
    base64_string = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAKT2lDQ1BQaG90b3Nob3AgSUNDIHByb2ZpbGUAAHjanVNnVFPpFj333vRCS4iAlEtvUhUIIFJCi4AUkSYqIQkQSoghodkVUcERRUUEG8igiAOOjoCMFVEsDIoK2AfkIaKOg6OIisr74Xuja9a89+bN/rXXPues852zzwfACAyWSDNRNYAMqUIeEeCDx8TG4eQuQIEKJHAAEAizZCFz/SMBAPh+PDwrIsAHvgABeNMLCADATZvAMByH/w/qQplcAYCEAcB0kThLCIAUAEB6jkKmAEBGAYCdmCZTAKAEAGDLY2LjAFAtAGAnf+bTAICd+Jl7AQBblCEVAaCRACATZYhEAGg7AKzPVopFAFgwABRmS8Q5ANgtADBJV2ZIALC3AMDOEAuyAAgMADBRiIUpAAR7AGDIIyN4AISZABRG8lc88SuuEOcqAAB4mbI8uSQ5RYFbCC1xB1dXLh4ozkkXKxQ2YQJhmkAuwnmZGTKBNA/g88wAAKCRFRHgg/P9eM4Ors7ONo62Dl8t6r8G/yJiYuP+5c+rcEAAAOF0ftH+LC+zGoA7BoBt/qIl7gRoXgugdfeLZrIPQLUAoOnaV/Nw+H48PEWhkLnZ2eXk5NhKxEJbYcpXff5nwl/AV/1s+X48/Pf14L7iJIEyXYFHBPjgwsz0TKUcz5IJhGLc5o9H/LcL//wd0yLESWK5WCoU41EScY5EmozzMqUiiUKSKcUl0v9k4t8s+wM+3zUAsGo+AXuRLahdYwP2SycQWHTA4vcAAPK7b8HUKAgDgGiD4c93/+8//UegJQCAZkmScQAAXkQkLlTKsz/HCAAARKCBKrBBG/TBGCzABhzBBdzBC/xgNoRCJMTCQhBCCmSAHHJgKayCQiiGzbAdKmAv1EAdNMBRaIaTcA4uwlW4Dj1wD/phCJ7BKLyBCQRByAgTYSHaiAFiilgjjggXmYX4IcFIBBKLJCDJiBRRIkuRNUgxUopUIFVIHfI9cgI5h1xGupE7yAAygvyGvEcxlIGyUT3UDLVDuag3GoRGogvQZHQxmo8WoJvQcrQaPYw2oefQq2gP2o8+Q8cwwOgYBzPEbDAuxsNCsTgsCZNjy7EirAyrxhqwVqwDu4n1Y8+xdwQSgUXACTYEd0IgYR5BSFhMWE7YSKggHCQ0EdoJNwkDhFHCJyKTqEu0JroR+cQYYjIxh1hILCPWEo8TLxB7iEPENyQSiUMyJ7mQAkmxpFTSEtJG0m5SI+ksqZs0SBojk8naZGuyBzmULCAryIXkneTD5DPkG+Qh8lsKnWJAcaT4U+IoUspqShnlEOU05QZlmDJBVaOaUt2ooVQRNY9aQq2htlKvUYeoEzR1mjnNgxZJS6WtopXTGmgXaPdpr+h0uhHdlR5Ol9BX0svpR+iX6AP0dwwNhhWDx4hnKBmbGAcYZxl3GK+YTKYZ04sZx1QwNzHrmOeZD5lvVVgqtip8FZHKCpVKlSaVGyovVKmqpqreqgtV81XLVI+pXlN9rkZVM1PjqQnUlqtVqp1Q61MbU2epO6iHqmeob1Q/pH5Z/YkGWcNMw09DpFGgsV/jvMYgC2MZs3gsIWsNq4Z1gTXEJrHN2Xx2KruY/R27iz2qqaE5QzNKM1ezUvOUZj8H45hx+Jx0TgnnKKeX836K3hTvKeIpG6Y0TLkxZVxrqpaXllirSKtRq0frvTau7aedpr1Fu1n7gQ5Bx0onXCdHZ4/OBZ3nU9lT3acKpxZNPTr1ri6qa6UbobtEd79up+6Ynr5egJ5Mb6feeb3n+hx9L/1U/W36p/VHDFgGswwkBtsMzhg8xTVxbzwdL8fb8VFDXcNAQ6VhlWGX4YSRudE8o9VGjUYPjGnGXOMk423GbcajJgYmISZLTepN7ppSTbmmKaY7TDtMx83MzaLN1pk1mz0x1zLnm+eb15vft2BaeFostqi2uGVJsuRaplnutrxuhVo5WaVYVVpds0atna0l1rutu6cRp7lOk06rntZnw7Dxtsm2qbcZsOXYBtuutm22fWFnYhdnt8Wuw+6TvZN9un2N/T0HDYfZDqsdWh1+c7RyFDpWOt6azpzuP33F9JbpL2dYzxDP2DPjthPLKcRpnVOb00dnF2e5c4PziIuJS4LLLpc+Lpsbxt3IveRKdPVxXeF60vWdm7Obwu2o26/uNu5p7ofcn8w0nymeWTNz0MPIQ+BR5dE/C5+VMGvfrH5PQ0+BZ7XnIy9jL5FXrdewt6V3qvdh7xc+9j5yn+M+4zw33jLeWV/MN8C3yLfLT8Nvnl+F30N/I/9k/3r/0QCngCUBZwOJgUGBWwL7+Hp8Ib+OPzrbZfay2e1BjKC5QRVBj4KtguXBrSFoyOyQrSH355jOkc5pDoVQfujW0Adh5mGLw34MJ4WHhVeGP45wiFga0TGXNXfR3ENz30T6RJZE3ptnMU85ry1KNSo+qi5qPNo3ujS6P8YuZlnM1VidWElsSxw5LiquNm5svt/87fOH4p3iC+N7F5gvyF1weaHOwvSFpxapLhIsOpZATIhOOJTwQRAqqBaMJfITdyWOCnnCHcJnIi/RNtGI2ENcKh5O8kgqTXqS7JG8NXkkxTOlLOW5hCepkLxMDUzdmzqeFpp2IG0yPTq9MYOSkZBxQqohTZO2Z+pn5mZ2y6xlhbL+xW6Lty8elQfJa7OQrAVZLQq2QqboVFoo1yoHsmdlV2a/zYnKOZarnivN7cyzytuQN5zvn//tEsIS4ZK2pYZLVy0dWOa9rGo5sjxxedsK4xUFK4ZWBqw8uIq2Km3VT6vtV5eufr0mek1rgV7ByoLBtQFr6wtVCuWFfevc1+1dT1gvWd+1YfqGnRs+FYmKrhTbF5cVf9go3HjlG4dvyr+Z3JS0qavEuWTPZtJm6ebeLZ5bDpaql+aXDm4N2dq0Dd9WtO319kXbL5fNKNu7g7ZDuaO/PLi8ZafJzs07P1SkVPRU+lQ27tLdtWHX+G7R7ht7vPY07NXbW7z3/T7JvttVAVVN1WbVZftJ+7P3P66Jqun4lvttXa1ObXHtxwPSA/0HIw6217nU1R3SPVRSj9Yr60cOxx++/p3vdy0NNg1VjZzG4iNwRHnk6fcJ3/ceDTradox7rOEH0x92HWcdL2pCmvKaRptTmvtbYlu6T8w+0dbq3nr8R9sfD5w0PFl5SvNUyWna6YLTk2fyz4ydlZ19fi753GDborZ752PO32oPb++6EHTh0kX/i+c7vDvOXPK4dPKy2+UTV7hXmq86X23qdOo8/pPTT8e7nLuarrlca7nuer21e2b36RueN87d9L158Rb/1tWeOT3dvfN6b/fF9/XfFt1+cif9zsu72Xcn7q28T7xf9EDtQdlD3YfVP1v+3Njv3H9qwHeg89HcR/cGhYPP/pH1jw9DBY+Zj8uGDYbrnjg+OTniP3L96fynQ89kzyaeF/6i/suuFxYvfvjV69fO0ZjRoZfyl5O/bXyl/erA6xmv28bCxh6+yXgzMV70VvvtwXfcdx3vo98PT+R8IH8o/2j5sfVT0Kf7kxmTk/8EA5jz/GMzLdsAAAAgY0hSTQAAeiUAAICDAAD5/wAAgOkAAHUwAADqYAAAOpgAABdvkl/FRgAADIVJREFUeNrkm3lUVfUWx/dlFOXKNSRLMwcGRVFE0MwUKpS0pWSOhWaAZik+REPM5Tw8S31iGWImmmDy0qdpEqgBqRAmzqIioCKiaJjcC4d5ut/3x7n3XI7nwB3kXl2rvdZvAef8hv37nP0b9v79ICIqWr5yOQAwT6bE478yRMR8t30b+6xRlUTyTgqYyBARQ0TMjFkzGADMa8NeY59JiHsnmiTEkBn7e6fOLzI3crLZepVglEolA4CprK5gerh0ZyysLJgNkeuZtDOnGHmpXKODUqhT8eNiRiqTatrgJ5CEioiImB9if4CYnMw4ASJC+MJw9oFSlZqRSQETQUQgCcHS2pL7nYhN7WXtMdR7KMb6j8Fwn+Fw6uUE6zbW3HsyZ39aWVkhLS2NV3fovH9p8qmSs6szfk9PRUsSFx+nKSMRJIYmTJrAVFdXixaWMyUgIrj2ddU8VLbYHiZ/OInfoOrnlqhvRPNfuHwBfd37CCAQEZKSktgPceok/70ZH4T7QHfczr/Nq1epUvTUqVMtA1CZi6goGAVXWF4qb7nnqgbzC/LR1rYtV66/e3+UlpVCm1zLuQo39768jtnY2KBYXowFCxcIvj5JCGTBt7Bvt25pok0jAOB4ynHDATSiEQ4vOoCIkHUti0dWHAH7bsPG9VyjV69fhT4SOj+U19E2bdvA3MIcRARbO1vI7O1gYWEhhKH6fdbsT3j1JackGw4AANzc3EBEiN8XrzMAAOjeozuICInHftWp4+ovxkLgj3e3fm7IvHgGZZWlqKqvQHbedWzbHo1Br3uhjY21AMKylUu5usLmzzMMgFqheWFsBb4jffX6kl9t+ApEBBdXFx0BKHkAp300lVN8d+zuZssVlxSjT78+giGSdysPAODq5tpc57UBYJX5+dDPXKX5BflarUAtNXU1sHewZ63gaCL0lXplPew72aNjRwed8r/hPZRnCW/5vgkAmD13NjuUbNqIgdA+BBRlcsg6yEBEmB3ymV6dCAz6GESEpSuWwhDJzctFVtYVnfOPencUzwri98Vj0eJFkEql6PBCB8MAAICnpydXaXpGus4KJSQmsMtoH1eYQpRQom+TlcThRQccPHIA/uPGGjYJqk097sdYrlIfHx+9lOr0cie87fc2TCXnLpzldLVuY41GNCBkbojhqwAANDQ24JVXu3AVR22N0lmhyqpKVNdU6Tx3tIYEBgdyun46ZxY8B3mKAyAdAQDAe+Pf4yq1lbZD0cMik3ZK1yEAABcuXuB0denlAp+3fYRLITsx6g5g1ZpVbAWqbWg/936oqqt6Vj3lJONMBnLycnivb+Rkcx1evOQLlDIKSCQSAQBPL0/dARxNTtJQVEEInBH4TAE8lj8GEWHqtKkAgLtFBYiLj0XnVzpzui5dsRT37heK7hp9fX11B1BaruAvJapKTmeefmYm/+91a0FEMLc0R3t7qeYrq9LEKRMBALcKb7J+g+p523asrxI2P0x3AADg85aPwIxe6PgCShQlgu1sq33pRnEPtLqmmvNTmks3crI1PsGJ37jnPx34L+L37UV6erpuANSTy8ZNG0RNycNrAK7lCZ2e8spyJB5LRMrvKahrqGuVMd9URo8ZrfE6PfqjV59e3N/fbPlaoP+lrEvYszeu6XP9LEDBKLjtbVMrICLsit0pyD9y1EjufUhoiN79vlNwBzm5OSJeagMAIDZOsz+ZOy8EdQ21CIsIw5XsKwKvtkFV5gmi+gEAgJC5c4RWYE7IvcVXdPWa1Txz7NKlC2obarVamrpz17Ovc2V37NwhapFpGZpgh6NzT62O1pMOl0EAyivL4dzLmde58PBwjrR6dpbJZDxIUqkUpUypzu14DPTgtRG9LVo0X2/X3iAiWFpaouhBkb5Gpj8ANVPn3k4C/0A9Ca5dt0YwTwz08tC59j0/7tGUt2TLj/AdIZo3aEYQ18bqNatNBQDwHekLIkLwzGDh8vTlWt7X6+/RH3n5eTrXvWPnDkH874sli0Tz7jvwE9dOUHCg6QC82u1VEBGSU5MFcwtTyWCY9zAQESK/jdS77hOqaLQ6DRri1Wze2/m3uHyDXxtsGgDZN9itpoenB7/zSohNNAbJlKlTQETwGuKJiqqKFvP27svOAx0dOqK8iuHNR0YBELMrBkSElatX6hTeMkTq6mvx4K8i1NbWas3bdB7IPJtpfAARiyNARDh2/Nhz4QVu/jqSA3D0+FHjD4GhqvjbpcuXDFb6XtE9/PmUfoTaxk5nZhi6EhgGoIdzTxARsrOzDVb+s9mfgYhaxQLybudqHKBJE4wLoL6hHjZSGxARih8VG6RwTX01unbvCiLC5i2RrQJh1qefsJPmYC/jAnj09yOOti5HXmISOo9/+nPl+mXDx4BqHGRmZoKI0K17N+MCUB9UtrNth4rKCr11fvT3I9jJ7Hg7xTlzZj+1BZSVlXHuuVEBlJaVcru0yupK/b9+aKjAm3xj+BvCr2rAKuri4gKJGRkXwOMSNgxl3c5Kq3f3pBTeuwtpe6moO516MlXUtPUGIJEYF4D6tNX7TW+9lDt/8TwXihIJT2P02NFPPQzenzBO35VFfwC/pbChJXcPd6151UGI8xfPcUfczRxScpYQsTQCigq5QQBmfjrT+ADS/0hjr7HYWEHByLWG0SprKuHo7Ci4MdIcACKCxFyCwwmHtUXEhF5kzPdo176tcQGUMWWcooX3C7Xv09WnNObCjmoDMS98nl4WsHPXTthKbY0LgClnYK26lJB6MqXFvBcvX+B1qNNLnfBb6nH8kngYDp0cxEE8cU9o7vwQ6GoKMbtiYGtrZAAA0MuVjb6uWLlCy+5sFtcRO1l73Lx1k3tXW1cDn7e8eYBs29tCZi/THL6ogISFh+luAaYAMOrdd0BEWLhoYYv5RviN4Dp38OBB0TwbN2tC7YcTDgEAAj7+UBDjj90Tq1WvyM2RsG5jbXwAq1az54Segz1bthRVnN7vHb8W86WcSEH6H/x7B3vi4zR7BiKYWZrhUtZFLfNNkPFXAQBIPZnKmemD4geiee7cvcMpfyQhwaBlrbauBoMGe2nO8ka0fM8geGZws0GaVgXQiAZ07cZ6c3E/xolvmJLZDZPnIM+niww11mGA1wAOwrov14nDqq/FlA8mIzc31zRB0aAgdnlbtnwZb91vOh6JCJsiDXd3lU0uX1q3s24x8Jmckgz/9/xNFxVW3/9xcnESKAwAERERLd4QO5l2AucunH3CshqbjSVGRUext0elNlCUKrhlUR37i/w6EluitpgOAAD06NkDRIT9/9svADDWn72YJGdKRMsOGTIERITJAZNwJOkXlFWUafWFOnd9GURCL/ThXw/h7eONOwV3TAtAfWtkwuTxgndOvZzwcueXRMudOfun6HH2kuVL8Nfjh81G/6K/i0bARwGCt3PmzoGfn58hXXg6AADg1o+9SrsxciPvORFhhJ/4cdahBPbyZdNlrmn6YNoHOkzErOknJBzh3Qw1OYDzl85xiv+icmAYhmGPxOeG8JRVy5at38DJmZ071m5YC8fePWFmbsaDkJKSorXth8UPQUSY9tE0Q9V/egAAEL09mlM848wfqG+oBxFhQcQCUQCbIjdhwkR+9Pbew0IEz2I3Mt0du6NSdRrU3PZfUaaAk5MjLC0tubmnUf8bKq0DAAC+WLKIva9jYY4eTuzkGLMrhpscG9HAKbpi9XJ8Hv65aD337hfi0d+PeKvCk6tD4rFEWFixV+aXr1iuaqPRkDBS6wEAgJjdO3hmPD14OmobawT5AgICEBoaqnf9Z86fxoCBmk1R1LZvobOraAoA7B5+Dw9CB/sOGD9lPKJ3bEXBffa2+dBhr2N64HS+JhVlYCoYlJWWgilnIC+V415RIW4W5CHh6BG49Xfj6nRyccSt/FutEUhtXQDqMagoU2DcxHGwUv/jlDrSI5HA0krzzNbOFv7v+2OM/xjIOshYV1jS/K2vLl264NDhQ+KrpBLPhwUIY4jH0cOxZ4vX2USTGUFmL4NdBzv4j/dH1tUsY6hnfABquZZ9Femn07Bk5RJIZbYYP+V9BH0SiJGjRmDo8Nex/j/r8fupVNwtKkBVbSUU5XI0NDYYWy3jAdDcEeDbZn19PR7LH+M5EdNZgFrKy8uReDQRtfW1/0wAe/fuxZyQkCZr9z8MQFJSEqKiorjAyj8OQOzuWGz//rvnC8CyZcuQkZFhkhb379+PAwcOiEaRngmAVatWFRERFi9eLPpv8a2dzp09x9y4wf57vBJKk7TZTEJJSUnR/wcAOL4eDthNrWcAAAAASUVORK5CYII="
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(BytesIO(imgdata))
    img = np.array(image)[:,:,0]/255.0
    print (img)
    plt.imshow(img)
    plt.show()

    l = np.reshape(img,[4096]).tolist()
    s = ""
    for e in l:
        s += str(e) + ", "

    f = open("D:/src/quickDraw/" + "1.txt", "w")
    f.write(s[:-2])
    f.close()


aclassDict = {0 : 'ant', 1 : 'bat', 2 : 'bear', 3 : 'bee', 4 : 'bird', 5 : 'butterfly', 6 : 'camel', 7 : 'cat', 8 : 'cow', 9 : 'crab', 10 : 'crocodile', 11 : 'dog', 12 : 'dolphin', 13 : 'dragon', 14 : 'duck', 15 : 'elephant', 16 : 'fish', 17 : 'flamingo', 18 : 'frog', 19 : 'giraffe', 20 : 'hedgehog', 21 : 'horse', 22 : 'kangaroo', 23 : 'lion', 24 : 'lobster', 25 : 'mermaid', 26 : 'monkey', 27 : 'mosquito', 28 : 'mouse', 29 : 'octopus', 30 : 'owl', 31 : 'panda', 32 : 'penguin', 33 : 'pig', 34 : 'rabbit', 35 : 'raccoon', 36 : 'rhinoceros', 37 : 'scorpion', 38 : 'sea turtle', 39 : 'shark', 40 : 'sheep', 41 : 'snail', 42 : 'snake', 43 : 'spider', 44 : 'squirrel', 45 : 'swan', 46 : 'tiger', 47 : 'whale', 48 : 'zebra'}

for a,b in aclassDict.items():
    print (b)



