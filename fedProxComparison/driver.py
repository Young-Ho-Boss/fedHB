import threading
import queue
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# FedAvg와 FedProx 스크립트에서 실행 함수 임포트
from fedAvg import run_fedavg
from fedProx import run_fedprox

class StreamRedirector:
    """
    stdout/stderr 스트림을 텍스트 위젯으로 리다이렉트
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        # 주기적으로 큐를 폴링
        self.text_widget.after(100, self.poll)

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass

    def poll(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.text_widget.configure(state='normal')
                self.text_widget.insert('end', msg)
                self.text_widget.see('end')
                self.text_widget.configure(state='disabled')
        except queue.Empty:
            pass
        # 100ms 후 재실행
        self.text_widget.after(100, self.poll)


def start_task(task_func, frame, title):
    """
    주어진 학습 함수를 백그라운드 스레드로 실행하고,
    로그와 결과 플롯을 지정된 frame에 표시
    """
    # 제목 레이블
    tk.Label(frame, text=title, font=('Arial', 12, 'bold')).pack(fill='x')
    # 로그 출력을 위한 ScrolledText
    text = ScrolledText(frame, state='disabled', height=20)
    text.pack(fill='both', expand=True)
    redirector = StreamRedirector(text)

    def task():
        # stdout 리다이렉트
        import sys, contextlib
        with contextlib.redirect_stdout(redirector):
            accs = task_func()
        # 완료 후 결과 플롯
        fig = plt.Figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(accs)+1), accs, marker='o')
        ax.set_title(f'{title} Accuracy')
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy (%)')
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill='both', expand=False)
        canvas.draw()

    threading.Thread(target=task, daemon=True).start()


def main():
    root = tk.Tk()
    root.title('Federated Learning GUI Driver')

    # 좌우 프레임 분할
    left = tk.Frame(root)
    right = tk.Frame(root)
    left.pack(side='left', fill='both', expand=True)
    right.pack(side='right', fill='both', expand=True)

    # FedAvg와 FedProx 동시에 실행
    start_task(run_fedavg, left, 'FedAvg')
    start_task(run_fedprox, right, 'FedProx')

    root.mainloop()

if __name__ == '__main__':
    main()
