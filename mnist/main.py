from tkinter import Tk, Canvas
from numpy import zeros, array
#from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils
false = False
true = True
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape(X_train.shape[0], -1) / 255.
#X_test = X_test.reshape(X_test.shape[0], -1) / 255.
#y_train = np_utils.to_categorical(y_train, num_classes=10)
#y_test = np_utils.to_categorical(y_test, num_classes=10)

def max_index(array):
    maxIndex = 0
    for i in range(0, len(array)):
        if array[i] > array[maxIndex]:
            maxIndex = i
    return maxIndex


class Window:
    isButtonPressed = false
    threadStop = false
    image = zeros([28, 28])

    def __init__(self, title, width, height):
        self.tk = Tk()
        self.tk.title(title)
        self.tk.geometry(f"{width}x{height}")
        self.tk.resizable(false, false)
        self.bind_event()

        self.title = title
        self.width = width
        self.height = height

        self.model = load_model("mnist.h5")
        self.create_component()
        self.tk.after(33, self.draw)
        self.tk.mainloop()

    def create_component(self):
        self.canvas = Canvas(self.tk, width=self.width, height=self.height)
        self.canvas.pack()

    def bind_event(self):
        self.tk.bind("<Button-1>", self.press_down_left_button)
        self.tk.bind("<ButtonRelease-1>", self.release_up_left_button)
        self.tk.bind("<KeyPress-c>", self.press_down_c_key)
        self.tk.bind("<Motion>", self.mouse_move)
        self.tk.bind("<KeyPress-p>", self.predict)
        self.tk.bind("<KeyPress-h>", self.hiden)

    def press_down_c_key(self, event):
        # 用来清理当前用户绘制的数字图像
        self.image = zeros([28, 28])
        self.canvas.delete('all')

    def press_down_left_button(self, event):
        self.isButtonPressed = true
        self.image[event.x // 10][event.y // 10] = 1.0

    def release_up_left_button(self, event):
        self.isButtonPressed = false

    def mouse_move(self, event):
        if self.isButtonPressed:
            self.image[event.x//10][event.y//10] = 1.0

    def draw(self):
        for x in range(0, 28):
            for y in range(0, 28):
                if self.image[x][y] != 0:
                    self.canvas.create_rectangle(10*x, 10*y, 10*(x+1), 10*(y+1))
        self.tk.after(33, self.draw)

    def predict(self, evnet):
        # transform image, because my system input data need transform to his form
        # 根据中间对称轴反转
        buf1 = zeros([28, 28])
        for x in range(0, 28):
            for y in range(0, 28):
                buf1[x][27-y] = self.image[x][y]
        # 逆时针旋转90度
        buf2 = zeros([28,28])
        for x in range(0, 28):
            for y in range(0, 28):
                buf2[27-y][x] = buf1[x][y]

        image = array([buf2.reshape(-1)])
        print(max_index(self.model.predict(image)[0]))

    def hiden(self, event):
        self.canvas.delete("all")
        self.image = X_train[2].reshape(28, 28)


if __name__ == "__main__":
    window = Window("mnist", 280, 280)







































