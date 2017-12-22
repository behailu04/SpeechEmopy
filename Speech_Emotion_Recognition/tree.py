try:
    from Tkinter import *
except:
    from tkinter import *
from math import *

deg = pi / 4
w = 0.01
right = False
paused = False
w_temp = 0.01


def speed_up(event):
    global w
    if w < 0.4:
        w += 0.01


def slow_down(event):
    global w
    if w > 0:
        w -= 0.01


def stop_resume(event):
    global w, w_temp, paused
    if (event.char == 's' or event.char == 'S') and not paused:
        w_temp = w
        w = 0
        paused = True
    elif event.char == 'r' or event.char == 'R':
        w = w_temp
        paused = False


pendulum_size = 300  # adjust pendulum size here
canvas_size = pendulum_size + 10
canvas = Canvas(width=canvas_size, height=canvas_size, bg='white')
canvas.bind('<Up>', speed_up)
canvas.bind('<Down>', slow_down)
canvas.bind('<Key>', stop_resume)
canvas.pack(expand=NO)


def swing():
    global deg, w, right
    canvas.create_oval(canvas_size / 2 - 2, 0.1 * canvas_size - 2, canvas_size / 2 + 2, 0.1 * canvas_size + 2,
                       fill="black")
    canvas.create_line(canvas_size / 2, 0.1 * canvas_size, canvas_size / 2 + 0.7 * pendulum_size * sin(deg),
                       0.1 * canvas_size + 0.7 * pendulum_size * cos(deg))
    canvas.create_oval(canvas_size / 2 + 0.7 * pendulum_size * sin(deg) - 0.05 * pendulum_size,
                       0.1 * canvas_size + 0.7 * pendulum_size * cos(deg) - 0.05 * pendulum_size,
                       canvas_size / 2 + 0.7 * pendulum_size * sin(deg) + 0.05 * pendulum_size,
                       0.1 * canvas_size + 0.7 * pendulum_size * cos(deg) + 0.05 * pendulum_size, fill="black")
    if right:
        deg += w
    else:
        deg -= w
    if deg > pi / 4.3:
        right = False
    elif deg < -pi / 4.3:
        right = True


def callback_fun():
    canvas.delete(ALL)
    swing()
    canvas.after(10, callback_fun)


canvas.focus_set()
callback_fun()

mainloop()
