from tkinter import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

CELL_WIDTH = 17
CANVAS_WIDTH = 28 * CELL_WIDTH

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=2, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=40*3*3, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 40*3*3)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)




class Cell:
    def __init__(self, x, y):
        self.bg = "#ffffff"
        self.x = x
        self.y = y

        self.prevx = self.prevy = 0

    def switch_color(self):
        self.bg = "#000000" if self.bg == "#ffffff" else "#000000"

class CCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cells = {}
        self.drawing = False

        self.initialize_lines()
        self.bind("<Button-1>", self.left_mouse_clicked)
        self.bind("<B1-Motion>", self.left_mouse_clicked)
        self.bind("<Button-2>", self.right_mouse_clicked)
        self.bind("<B2-Motion>", self.right_mouse_clicked)


    def initialize_lines(self):
        for i in range(28):
            for j in range(28):
                c = Cell(i*CELL_WIDTH, j*CELL_WIDTH)
                self.cells[(i, j)] = c
                self.create_rectangle(c.x, c.y, c.x+CELL_WIDTH, c.y+CELL_WIDTH, fill=c.bg, width=0, tags=f"cell({i},{j})")


    def left_mouse_clicked(self, e=None):
        self.drawing = True
        ex, ey = e.x // CELL_WIDTH, e.y // CELL_WIDTH

        self.cells[(ex, ey)].switch_color()

        self.itemconfig(f"cell({ex},{ey})", fill="#000000")

    def left_mouse_drag(self, e=None):
        if not self.drawing: return
        ex, ey = e.x // CELL_WIDTH, e.y // CELL_WIDTH

        if ex == self.prevx and ey == self.prevy: return
        self.prevx = ex
        self.prevy = ey

        self.cells[(ex, ey)].switch_color()
        
        self.itemconfig(f"cell({ex},{ey})", fill="#000000")

    def right_mouse_clicked(self, e=None):
        self.drawing = True
        ex, ey = e.x // CELL_WIDTH, e.y // CELL_WIDTH

        self.cells[(ex, ey)].switch_color()

        self.itemconfig(f"cell({ex},{ey})", fill="#ffffff")

    def right_mouse_drag(self, e=None):
        if not self.drawing: return
        ex, ey = e.x // CELL_WIDTH, e.y // CELL_WIDTH

        if ex == self.prevx and ey == self.prevy: return
        self.prevx = ex
        self.prevy = ey

        self.cells[(ex, ey)].switch_color()
        
        self.itemconfig(f"cell({ex},{ey})", fill="#ffffff")

    def clear(self, e=None):
        self.delete("all")
        self.initialize_lines()


def predict(win, canvas: CCanvas, olbl):
    data = [canvas.cells[(j, i)].bg == "#000000" for i in range(28) for j in range(28)]
    data = torch.Tensor(data).reshape(1, 28, 28)

    loaded_model_0 = CNN()

    loaded_model_0.load_state_dict(torch.load(f="Models/01_model_0.pth"))

    loaded_model_0.eval()
    with torch.inference_mode():
        loaded_model_preds: torch.Tensor = loaded_model_0(data)
        predictions = loaded_model_preds.tolist()[0]

    prediction = predictions.index(max(predictions))

    print(predictions, prediction)


if __name__ == '__main__':
    win = Tk()
    win.geometry(f"{CANVAS_WIDTH + 200}x{CANVAS_WIDTH}")

    canvas = CCanvas(win, bg="#ffffff", bd=0, highlightthickness=0)
    canvas.place(x=0, y=0, width=CANVAS_WIDTH, relheight=1)

    f1 = Frame(win, bg="#222222")
    f1.place(x=CANVAS_WIDTH, y=0, width=200, relheight=1)

    clear = Button(f1, text="Clear")
    clear.place(x=10, y=10, height=50, width=180)
    clear.bind("<Button-1>", canvas.clear)

    output_lbl = Label(f1, text="", bg="#ffffff")
    output_lbl.place(x=10, y=190, height=50, width=180)

    predict_btn = Button(f1, text="Predict")
    predict_btn.place(x=10, y=60, height=50, width=180)
    predict_btn.bind("<Button-1>", lambda x: predict(win, canvas, output_lbl))

    win.mainloop()