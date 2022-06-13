#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tkinter as tk
import os
from tkinter import messagebox, filedialog, dialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# In[2]:


# basic variables
# range of work space
work_space_width = 660
work_space_height = 608
# size of matrices
mat_size = 3
# size of output words
word_size = 16
# size of output numbers
number_size = 12
# size of output brackets
bracket_size = 50


# ## -----矩陣計算機-----

# ### -----Basic Functions-----

# In[3]:


def check_mat_inputs(mat_input):
    # error happened(True), didn't happen(False)
    error = False
    for i in range(mat_size):
        for j in range(mat_size):
            # default(0) of no input
            if mat_input[i*mat_size+j].get().rstrip() == '' :
                mat_input[i*mat_size+j].delete(0, 'end')
                mat_input[i*mat_size+j].insert(0, 0)
            # check inputs are numbers
            try:
                element = float(mat_input[i*mat_size+j].get())
            except:
                error = True
                messagebox.showerror('錯誤', '請輸入數字')                
    return error

def check_number_input(num_input):
    # error happened(True), didn't happen(False)
    error = False
    try:
        number = float(num_input.get())
    except:
        error = True
        num_input.delete(0, 'end')
        num_input.insert(0, 2)
        number = 2
        messagebox.showerror('錯誤', '請輸入數字\n提供預設值：2')
        
    return number, error

def determine_int_float(num_input):
    if int(num_input)-num_input == 0:
        result = int(num_input)
    else:
        result = num_input
    return result

def get_mat_value(mat_input):
    result_mat = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(mat_size):
            result_mat[i][j] = float(mat_input[i*mat_size+j].get())
    
    return result_mat

def recreate_result_frame():
    # destroy widgets
    for widget in result_frame.winfo_children():
        widget.destroy()

def show_calculation_result_number(input_number, pos):
    # show number
    number = tk.Label(result_frame, text=str(input_number), font=('Arial', number_size))
    number.grid(row=0, column=pos, pady=68, sticky='nw')
    pos += 1
    return pos
        
def show_calculation_result_matrix(mat_input, pos):
    # show matrix  
    mat_brackets = tk.Label(result_frame, text='(', font=('Arial', bracket_size))
    mat_brackets.grid(row=0, column=pos, pady=35, sticky='nw')
    pos += 1
    mat_text = ''
    for i in range(mat_size):
        for j in range(mat_size):
            element = mat_input[i][j]
            # integeal cases
            if int(element)-element == 0:
                mat_text += '{0:^7}'.format(str(int(element)))
            # float cases
            else:
                mat_text += '{0:^7}'.format(str(element))
        if i != mat_size-1:
            mat_text += '\n\n'
    mat_values = tk.Label(result_frame, text=mat_text, font=('Arial', number_size))
    mat_values.grid(row=0, column=pos, pady=30, sticky='nw')
    pos += 1
    mat_brackets = tk.Label(result_frame, text=')', font=('Arial', bracket_size))
    mat_brackets.grid(row=0, column=pos, pady=35, sticky='nw')
    pos += 1
    return pos

def show_calculation_result_determinant(mat_input, pos):
    # show matrix  
    mat_brackets = tk.Label(result_frame, text='︱', font=('Arial', bracket_size))
    mat_brackets.grid(row=0, column=pos, pady=35, sticky='nw')
    pos += 1
    mat_text = ''
    for i in range(mat_size):
        for j in range(mat_size):
            element = mat_input[i][j]
            # integeal cases
            if int(element)-element == 0:
                mat_text += '{0:^7}'.format(str(int(element)))
            # float cases
            else:
                mat_text += '{0:^7}'.format(str(element))
        if i != mat_size-1:
            mat_text += '\n\n'
    mat_values = tk.Label(result_frame, text=mat_text, font=('Arial', number_size))
    mat_values.grid(row=0, column=pos, pady=30, sticky='nw')
    pos += 1
    mat_brackets = tk.Label(result_frame, text='︱', font=('Arial', bracket_size))
    mat_brackets.grid(row=0, column=pos, pady=35, sticky='nw')
    pos += 1
    return pos
    
def show_calculation_result_symbol(mode, pos):
    # show operation symbol
    if mode == 'mul':
        symbol = tk.Label(result_frame, text='‧', font=('Arial', word_size))
    elif mode == 'add':
        symbol = tk.Label(result_frame, text='+', font=('Arial', word_size))
    elif mode == 'sub':
        symbol = tk.Label(result_frame, text='-', font=('Arial', word_size))
    elif mode == 'eq':
        symbol = tk.Label(result_frame, text='=', font=('Arial', word_size))
    symbol.grid(row=0, column=pos, pady=66, sticky='nw')
    pos += 1
    return pos

def show_calculation_result_sign(sign, pos):
    # show sign
    if sign == 'trans':
        upper_index = tk.Label(result_frame, text='𝙏', font=('Arial', word_size))
    elif sign == 'inv':
        upper_index = tk.Label(result_frame, text='(-1)', font=('Arial', word_size))
    upper_index.grid(row=0, column=pos, pady=32, sticky='nw')
    pos += 1
    return pos

def show_calculation_result_words(string_input, pos):
    words = tk.Label(result_frame, text=string_input, font=('Arial', word_size))
    words.grid(row=0, column=pos, padx=5, pady=66, sticky='nw')
    pos += 1
    return pos

def mat_mul(mat_A, mat_B):
    result_mat = np.matmul(mat_A, mat_B)
    
    return result_mat

def mat_add(mat_A, mat_B):
    result_mat = np.add(mat_A, mat_B)
    
    return result_mat

def mat_sub(mat_A, mat_B):
    result_mat = np.add(mat_A, -1*mat_B)
    
    return result_mat

def mat_trans(mat_input):
    result_mat = np.transpose(mat_input)
    
    return result_mat

def find_det(mat_input):
    result = np.linalg.det(mat_input)
    result = round(result, 3)
    # integeal cases
    if int(result)-result == 0:
        result = int(result)
        
    return result

def find_inv(mat_input):
    result_mat = np.linalg.inv(mat_input)
    for i in range(mat_size):
        for j in range(mat_size):
            result_mat[i][j] = round(result_mat[i][j], 3)
            # integeal cases
            if int(result_mat[i][j])-result_mat[i][j] == 0:
                result_mat[i][j] = int(result_mat[i][j])
    
    return result_mat

def find_rank(mat_input):
    result = np.linalg.matrix_rank(mat_input)
    
    return result

def multiply_number(mat_input, multiple):
    result_mat = multiple*mat_input
    
    return result_mat

def LU_decomposition(mat_input):
    l_matrix = np.zeros((mat_size, mat_size))
    mat_tem = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(mat_size):
            mat_tem[i][j] = mat_input[i][j]
    
    for row in range(mat_size):
        for minus in range(row, mat_size-1): # 用以記錄相減位置
            if mat_tem[minus+1][row] != 0:
                multiple = mat_tem[minus+1][row] / mat_tem[row][row] # 相減倍數
                # 處理 L_matrix
                l_matrix[minus+1][row] = multiple
                
                for index in range(row, len(np.flip(mat_tem))):
                    # 處理矩陣A
                    mat_tem[minus+1][index] -= mat_tem[row][index] * multiple
    # U_matrix
    u_matrix = mat_tem
    # L_matrix
    # L_matrix之對角線值為 1
    for pos in range(len(l_matrix)):
        l_matrix[pos][pos] = 1
    
    return l_matrix, u_matrix


# ### -----Click Events-----

# In[56]:


def mat_cal_info_button_click():
    messagebox.showinfo('提示', '請輸入3×3之矩陣')

def mat_exch_button_click(mat_A, mat_B):
    # check input
    error_A = check_mat_inputs(mat_A)
    error_B = check_mat_inputs(mat_B)
    
    if not (error_A or error_B):
        # matrix T for storing values of matrix A
        mat_T = []
        for i in range(mat_size):
            for j in range(mat_size):
                mat_T.append(mat_A[i*mat_size+j].get())
                mat_A[i*mat_size+j].delete(0, 'end')
                mat_A[i*mat_size+j].insert(0, mat_B[i*mat_size+j].get())
                mat_B[i*mat_size+j].delete(0, 'end')
                mat_B[i*mat_size+j].insert(0, mat_T[i*mat_size+j])

def mat_mul_button_click(mat_A, mat_B):
    # check input
    error_A = check_mat_inputs(mat_A)
    error_B = check_mat_inputs(mat_B)
    
    if not (error_A or error_B):
        # get values of matrices
        matrix_A = get_mat_value(mat_A)
        matrix_B = get_mat_value(mat_B)
        # calculate the A × B
        result_mat = mat_mul(matrix_A, matrix_B)
        # show results
        pos = 0
        recreate_result_frame()
        pos = show_calculation_result_matrix(matrix_A, pos)
        pos = show_calculation_result_symbol('mul', pos)
        pos = show_calculation_result_matrix(matrix_B, pos)
        pos = show_calculation_result_symbol('eq', pos)
        pos = show_calculation_result_matrix(result_mat, pos)
    
def mat_add_button_click(mat_A, mat_B):
    # check input
    error_A = check_mat_inputs(mat_A)
    error_B = check_mat_inputs(mat_B)
    
    if not (error_A or error_B):
        # get values of matrices
        matrix_A = get_mat_value(mat_A)
        matrix_B = get_mat_value(mat_B)
        # calculate the A + B
        result_mat = mat_add(matrix_A, matrix_B)
        # show results
        pos = 0
        recreate_result_frame()
        pos = show_calculation_result_matrix(matrix_A, pos)
        pos = show_calculation_result_symbol('add', pos)
        pos = show_calculation_result_matrix(matrix_B, pos)
        pos = show_calculation_result_symbol('eq', pos)
        pos = show_calculation_result_matrix(result_mat, pos)
    
def mat_sub_button_click(mat_A, mat_B):
    # check input
    error_A = check_mat_inputs(mat_A)
    error_B = check_mat_inputs(mat_B)
    
    if not (error_A or error_B):
        # get values of matrices
        matrix_A = get_mat_value(mat_A)
        matrix_B = get_mat_value(mat_B)
        # calculate the A - B
        result_mat = mat_sub(matrix_A, matrix_B)
        # show results
        pos = 0
        recreate_result_frame()
        pos = show_calculation_result_matrix(matrix_A, pos)
        pos = show_calculation_result_symbol('sub', pos)
        pos = show_calculation_result_matrix(matrix_B, pos)
        pos = show_calculation_result_symbol('eq', pos)
        pos = show_calculation_result_matrix(result_mat, pos)
    
def trans_button_click(mat_input):
    # check input
    error = check_mat_inputs(mat_input)
    
    if not error:
        # get values of the matrix
        matrix = get_mat_value(mat_input)
        # transpose matrix
        result_mat = mat_trans(matrix)
        # show results
        pos = 0
        recreate_result_frame()
        pos = show_calculation_result_matrix(matrix, pos)
        pos = show_calculation_result_sign('trans', pos)
        pos = show_calculation_result_symbol('eq', pos)
        pos = show_calculation_result_matrix(result_mat, pos)
        
def find_det_button_click(mat_input):
    # check input
    error = check_mat_inputs(mat_input)
    
    if not error:
        # get values of the matrix
        matrix = get_mat_value(mat_input)
        # find determinant
        determinant = find_det(matrix)
        # show results
        pos = 0
        recreate_result_frame()
        pos = show_calculation_result_determinant(matrix, pos)
        pos = show_calculation_result_symbol('eq', pos)
        pos = show_calculation_result_number(determinant, pos)
        
def find_inv_button_click(mat_input):
    # check input
    error = check_mat_inputs(mat_input)
    
    if not error:
        # get values of the matrix
        matrix = get_mat_value(mat_input)
        # find determinant
        determinant = find_det(matrix)
        # inverse exists or not
        # exists
        if determinant == 0:
            messagebox.showerror('錯誤', 'The determinant is 0, the matrix is not invertible')
        else:
            result_mat = find_inv(matrix)
            # show results
            pos = 0
            recreate_result_frame()
            pos = show_calculation_result_matrix(matrix, pos)
            pos = show_calculation_result_sign('inv', pos)
            pos = show_calculation_result_symbol('eq', pos)
            pos = show_calculation_result_matrix(result_mat, pos)
            
def find_rank_button_click(mat_input):
    # check input
    error = check_mat_inputs(mat_input)
    
    if not error:
        # get values of the matrix
        matrix = get_mat_value(mat_input)
        # find the rank
        rank = find_rank(matrix)
        # show reesults
        pos = 0
        recreate_result_frame()
        pos = show_calculation_result_words('rank', pos)
        pos = show_calculation_result_matrix(matrix, pos)
        pos = show_calculation_result_symbol('eq', pos)
        pos = show_calculation_result_number(rank, pos)
        
def multiply_button_click(mat_input, mul):
    # check input
    error_m = check_mat_inputs(mat_input)
    multiple, error_n = check_number_input(mul)
    multiple = determine_int_float(multiple)
    
    if not (error_m or error_n):
        # get values of the matrix
        matrix = get_mat_value(mat_input)
        # calculating by multiple
        result_mat = multiply_number(matrix, multiple)
        # show results
        pos = 0
        recreate_result_frame()
        pos = show_calculation_result_number(multiple, pos)
        pos = show_calculation_result_symbol('mul', pos)
        pos = show_calculation_result_matrix(matrix, pos)
        pos = show_calculation_result_symbol('eq', pos)
        pos = show_calculation_result_matrix(result_mat, pos)

def find_LU_decomp_click(mat_input):
    # check input
    error = check_mat_inputs(mat_input)
    
    if not error:
        # get values of the matrix
        matrix = get_mat_value(mat_input)
        # operating LU decomposition
        L_matrix, U_matrix = LU_decomposition(matrix)
        # show results
        pos = 0
        recreate_result_frame()
        pos = show_calculation_result_matrix(matrix, pos)
        pos = show_calculation_result_symbol('eq', pos)
        pos = show_calculation_result_matrix(L_matrix, pos)
        pos = show_calculation_result_symbol('mul', pos)
        pos = show_calculation_result_matrix(U_matrix, pos)


# ### -----Widgets-----

# In[78]:


def gen_mat_cal_widgets(win):    
    # -----frames-----
    #  ---matrix A---
    mat_A_frame = tk.Frame(win, width=150, height=300)
    mat_A_frame.grid(row=0, column=0, padx=0, pady=30, sticky='nw')
    mat_A_frame.grid_propagate(0) # fix the frame size
    #  ---matrix B---
    mat_B_frame = tk.Frame(win, width=150, height=300)
    mat_B_frame.grid(row=0, column=2, padx=0, pady=30, sticky='nw')
    mat_B_frame.grid_propagate(0) # fix the frame size
    # ---control buttons---
    ctrl_button_frame = tk.Frame(win, width=75, height=150)
    ctrl_button_frame.grid(row=0, column=1, padx=0, pady=65, sticky='nw')
    ctrl_button_frame.grid_propagate(0) # fix the frame size
    # ---information button---
    info_button_frame = tk.Frame(win, width=5, height=5)
    info_button_frame.grid(row=0, column=2, padx=5, sticky='ne')
    
    # -----inputs-----
    #  ---matrix A---
    #  label
    mat_A_label = tk.Label(mat_A_frame, text='Matrix A:', font=('Arial', word_size))
    mat_A_label.grid(row=0, column=0, columnspan=3, sticky='w')
    #  entry
    mat_A = list()
    for i in range(mat_size):
        for j in range(mat_size):
            #element = tk.Text(mat_A_frame, width=5, height=1)
            element = tk.Entry(mat_A_frame, width=5)
            element.grid(row=1+i, column=1+j, padx=5, pady=5, sticky='nw')
            #element.configure(wrap=None)
            mat_A.append(element)
    #  ---matrix B---
    #  label
    mat_B_label = tk.Label(mat_B_frame, text='Matrix B:', font=('Arial', word_size))
    mat_B_label.grid(row=0, column=0, columnspan=3, sticky='w')
    #  entry
    mat_B = list()
    for i in range(mat_size):
        for j in range(mat_size):
            #element = tk.Text(mat_B_frame, width=5, height=1)
            element = tk.Entry(mat_B_frame, width=5)
            element.grid(row=1+i, column=1+j, padx=5, pady=5, sticky='nw')
            #element.configure(wrap=None)
            mat_B.append(element)
    # ---control buttons---
    pos_row = 0
    #-matrix exchange button-
    mat_exch_button = tk.Button(ctrl_button_frame, text='🠒\n🠐', width=2, height=3, bg='white', command=lambda: mat_exch_button_click(mat_A, mat_B))
    mat_exch_button.grid(row=pos_row, column=0, padx=0, pady=0, sticky='n')
    pos_row += 1
    #-matrix multipler button-
    mat_mul_button = tk.Button(ctrl_button_frame, text='A × B', width=6, height=1, bg='white', command=lambda: mat_mul_button_click(mat_A, mat_B))
    mat_mul_button.grid(row=pos_row, column=0, padx=0, pady=1, sticky='nw')
    pos_row += 1
    #-matrix adder button-
    mat_add_button = tk.Button(ctrl_button_frame, text='A + B', width=6, height=1, bg='white', command=lambda: mat_add_button_click(mat_A, mat_B))
    mat_add_button.grid(row=pos_row, column=0, padx=0, pady=1, sticky='nw')
    pos_row += 1
    #-matrix subtractor button-
    mat_sub_button = tk.Button(ctrl_button_frame, text='A - B', width=6, height=1, bg='white', command=lambda: mat_sub_button_click(mat_A, mat_B))
    mat_sub_button.grid(row=pos_row, column=0, padx=0, pady=1, sticky='nw')
    pos_row += 1
    #-find determinant button-
    # matrix A
    A_find_det_button = tk.Button(mat_A_frame, text='Find the determinant', width=20, height=1, bg='white', command=lambda: find_det_button_click(mat_A))
    A_find_det_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    # matrix B
    B_find_det_button = tk.Button(mat_B_frame, text='Find the determinant', width=20, height=1, bg='white', command=lambda: find_det_button_click(mat_B))
    B_find_det_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    pos_row += 1
    #-find inverse button-
    # matrix A
    A_find_inv_button = tk.Button(mat_A_frame, text='Find the inverse', width=20, height=1, bg='white', command=lambda: find_inv_button_click(mat_A))
    A_find_inv_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    # matrix B
    B_find_det_button = tk.Button(mat_B_frame, text='Find the inverse', width=20, height=1, bg='white', command=lambda: find_inv_button_click(mat_B))
    B_find_det_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    pos_row += 1
    #-transpose button-
    # matrix A
    A_trans_button = tk.Button(mat_A_frame, text='Transpose', width=20, height=1, bg='white', command=lambda: trans_button_click(mat_A))
    A_trans_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    # matrix B
    B_trans_button = tk.Button(mat_B_frame, text='Transpose', width=20, height=1, bg='white', command=lambda: trans_button_click(mat_B))
    B_trans_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    pos_row += 1
    #-find rank button-
    # matrix A
    A_find_rank_button = tk.Button(mat_A_frame, text='Find the rank', width=20, height=1, bg='white', command=lambda: find_rank_button_click(mat_A))
    A_find_rank_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    # matrix B
    B_find_rank_button = tk.Button(mat_B_frame, text='Find the rank', width=20, height=1, bg='white', command=lambda: find_rank_button_click(mat_B))
    B_find_rank_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    pos_row += 1
    #-multiply button-
    # matrix A
    A_multiply_button = tk.Button(mat_A_frame, text='Multiply by', width=20, height=1, bg='white')
    A_multiply_button.grid(row=pos_row, column=0, columnspan=5, ipadx=0, pady=1, sticky='w')
    A_multiply_multiple = tk.Entry(mat_A_frame, width=2)
    A_multiply_multiple.grid(row=pos_row, column=3, sticky='e')
    A_multiply_button.config(command=lambda: multiply_button_click(mat_A, A_multiply_multiple))
    # matrix B
    B_multiply_button = tk.Button(mat_B_frame, text='Multiply by', width=20, height=1, bg='white')
    B_multiply_button.grid(row=pos_row, column=0, columnspan=5, ipadx=0, pady=1, sticky='w')
    B_multiply_multiple = tk.Entry(mat_B_frame, width=2)
    B_multiply_multiple.grid(row=pos_row, column=3, sticky='e')
    B_multiply_button.config(command=lambda: multiply_button_click(mat_B, B_multiply_multiple))
    pos_row += 1
    #-LU decomposition button-
    # matrix A
    A_LU_decomp_button = tk.Button(mat_A_frame, text='LU-decomposition', width=20, height=1, bg='white', command=lambda: find_LU_decomp_click(mat_A))
    A_LU_decomp_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    # matrix B
    B_LU_decomp_button = tk.Button(mat_B_frame, text='LU-decomposition', width=20, height=1, bg='white', command=lambda: find_LU_decomp_click(mat_B))
    B_LU_decomp_button.grid(row=pos_row, column=0, columnspan=10, ipadx=0, pady=1, sticky='n')
    pos_row += 1
    # ---information button---
    info_button = tk.Button(info_button_frame, text='提示', width=3, height=1, bg='white', command=lambda :mat_cal_info_button_click())
    info_button.grid(row=0, column=0, sticky='nw')
    
    # ---results---
    global result_frame
    result_frame = tk.LabelFrame(win, text='運算結果', width=work_space_width-8, height=200)
    result_frame.grid(row=6, column=0, columnspan=3, padx=2, sticky='nw')
    result_frame.grid_propagate(0) # fix the frame size


# ## -----資料作圖機-----

# ### -----basic functions-----

# In[79]:


def get_data(file_path):
    dataFile = open(file_path, 'r')
    x_value = []
    y_value = []
    dataFile.readline()  # discard header
    for line in dataFile:
        x, y = line.split()
        x_value.append(float(x))
        y_value.append(float(y))
    dataFile.close()
    return (x_value, y_value)
    
def show_file_path(file_path):
    # reset load text
    load_text.set('')
    
    text = load_text.get()
    text += file_path
    load_text.set(text)

def create_toolbar(win, canvas):
    toolbarFrame = tk.Frame(win)
    toolbarFrame.grid(row=2, column=1, padx=20, pady=170, sticky='sw')
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

def plot_figure(win, mode, plot_figure_setup, canvas):
    # clean the figure
    plot_figure_setup.clear()
    x, y = get_data(file_path)
    if mode == 'points':
        plot_figure_setup.plot(x, y, 'o')
    elif mode == 'line':
        plot_figure_setup.plot(x, y)
    elif mode == 'bar':
        plot_figure_setup.bar(x, y)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, rowspan=5, column=1, padx=20, sticky='nw')
    # creating the toolbar
    create_toolbar(win, canvas)
    
    # placing the toolbar on the Tkinter window
    canvas._tkcanvas.grid(row=0, rowspan=5, column=1)


# ### -----Click Events-----

# In[110]:


def plot_fig_info_button_click():
    messagebox.showinfo('提示', '請先選擇檔案(.txt)，再按下想要繪製的資料圖種類\n註：檔案內之資料形式須為以下格式：\n x                y\n (data x_1) (data y_1)\n (data x_2) (data y_2)\n')

def load_file_button_click():
    global file_path
    # file type
    file_type = (('text files', '*.txt'), ('All files', '*.*'))
    # file path
    file_path = filedialog.askopenfilename(title='選擇檔案', initialdir='/', filetypes=file_type)
    # show file path
    show_file_path(file_path)

def plot_data_points_button_click(win, plot_figure_setup, canvas):    
    plot_figure(win, 'points', plot_figure_setup, canvas)
    
def plot_line_chart_button_click(win, plot_figure_setup, canvas):
    plot_figure(win, 'line', plot_figure_setup, canvas)
    
def plot_bar_chart_button_click(plot_figure_frame, plot_figure_setup, canvas):
    plot_figure(win, 'bar', plot_figure_setup, canvas)


# ### -----Widgets-----

# In[111]:


def gen_data_fig_widgets(win):
    # -----frames-----
    # ---select file---
    global load_file_frame
    load_file_frame = tk.Frame(win, width=620, height=60)
    load_file_frame.grid(row=0, column=0, padx=0, pady=0, sticky='nw')
    load_file_frame.grid_propagate(0) # fix the frame size
    # ---plot figure---
    plot_figure_frame = tk.Frame(win, width=600, height=525)
    plot_figure_frame.grid(row=1, column=0, padx=0, pady=0, sticky='nw')
    plot_figure_frame.grid_propagate(0) # fix the frame size
    # ---information button---
    info_button_frame = tk.Frame(win, width=5, height=5)
    info_button_frame.grid(row=0, column=1, padx=1, sticky='ne')
    
    # -----figure-----
    # the figure that will contain the plot
    figure = Figure(figsize = (5, 4.8))
    plot_figure_setup = figure.add_subplot(111)
    # creating the Tkinter canvas
    canvas = FigureCanvasTkAgg(figure, plot_figure_frame)  
    canvas.draw()
    # placing the canvas on the window
    #canvas.get_tk_widget().grid(row=0, rowspan=5, column=1, sticky='nw')
    
    # -----control buttons-----
    #    ---load file---
    load_file_button = tk.Button(load_file_frame, text='選擇檔案', width=8, height=1, bg='white', font=('Arial', 10), command=lambda: load_file_button_click())
    load_file_button.grid(row=0, column=0, padx=2, pady=0, sticky='nw')
    #   ---plot figure---
    # -plot data points-
    plot_data_points_button = tk.Button(plot_figure_frame, text='繪\n製\n資\n料\n點', width=3, height=9, bg='white', font=('Arial', 10), command=lambda: plot_data_points_button_click(plot_figure_frame, plot_figure_setup, canvas))
    plot_data_points_button.grid(row=0, column=0, padx=2, pady=1, sticky='nw')
    # -plot line chart-
    plot_line_chart_button = tk.Button(plot_figure_frame, text='繪\n製\n折\n線\n圖', width=3, height=9, bg='white', font=('Arial', 10), command=lambda: plot_line_chart_button_click(plot_figure_frame, plot_figure_setup, canvas))
    plot_line_chart_button.grid(row=1, column=0, padx=2, pady=1, sticky='nw')
    # -plot bar chart-
    plot_bar_chart_button = tk.Button(plot_figure_frame, text='繪\n製\n直\n方\n圖', width=3, height=9, bg='white', font=('Arial', 10), command=lambda: plot_bar_chart_button_click(plot_figure_frame, plot_figure_setup, canvas))
    plot_bar_chart_button.grid(row=2, column=0, padx=2, pady=1, sticky='nw')
    # ---information button---
    info_button = tk.Button(info_button_frame, text='提示', width=3, height=1, bg='white', command=lambda :plot_fig_info_button_click())
    info_button.grid(row=0, column=0, sticky='nw')
    
    # -----label-----
    global load_text
    load_text = tk.StringVar()
    load_file_path_label = tk.Label(load_file_frame, textvariable=load_text, width=82, height=1, font=('Arial', 10))
    load_file_path_label.grid(row=2, column=0, padx=2, pady=0, sticky='nw')


# ## -----Main Page-----

# In[112]:


def mat_cal_button_click(win):
    # -----work space-----
    mat_cal_frame = tk.LabelFrame(win, text='矩陣計算機', width=work_space_width, height=work_space_height)
    mat_cal_frame.grid(row=0, rowspan=2, column=1, padx=3, pady=0, sticky='nw')
    mat_cal_frame.grid_propagate(0) # fix the frame size
    
    # generate widgets
    gen_mat_cal_widgets(mat_cal_frame)
    
    
def data_fig_button_click(win):
    # work sapce
    data_fig_frame = tk.LabelFrame(win, text='資料作圖機', width=work_space_width, height=work_space_height)
    data_fig_frame.grid(row=0, rowspan=2, column=1, padx=3, pady=0, sticky='nw')
    data_fig_frame.grid_propagate(0) # fix the frame size
    
    # generate widgets
    gen_data_fig_widgets(data_fig_frame)


# In[113]:


# window design
win = tk.Tk()
win.title('')
win.geometry('700x610')
win.resizable(False, False)

# -----功能選擇-----
#   ---矩陣計算---
mat_cal = tk.Frame(win, width=3, height=win.winfo_height()//2)
mat_cal.grid(row=0, column=0, padx=0, pady=1, sticky='nw')

mat_cal_button = tk.Button(mat_cal, text='矩\n陣\n計\n算\n機', width=3, height=16, font=('Arial', 12), command=lambda: mat_cal_button_click(win))
mat_cal_button.grid(row=0, column=0, sticky='nw')

#   ---資料作圖---
data_fig = tk.Frame(win, width=3, height=win.winfo_height()//2)
data_fig.grid(row=1, column=0, padx=0, pady=1, sticky='nw')

data_fig_button = tk.Button(data_fig, text='資\n料\n作\n圖\n機', width=3, height=16, font=('Arial', 12), command=lambda: data_fig_button_click(win))
data_fig_button.grid(row=0, column=0, sticky='nw')


# In[114]:


win.mainloop()


# In[ ]:




