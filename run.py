import facecog as fc

detector = fc.detectId(device='cpu')
detector.live()
