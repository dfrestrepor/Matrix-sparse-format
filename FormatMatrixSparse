import numpy as np
from scipy.sparse import random, bsr_matrix

FORMAT = { 'bsr': 'Sparse BLAS BSR Matrix Storage Format',
        'coo': 'Sparse BLAS Coordinate Matrix Storage Format',
        'csr': 'Sparse BLAS CSR Matrix Storage Format',
        'csc': 'Sparse BLAS CSC Matrix Storage Format'
}

class FormatMatrixSparse(object):
    '''
    FormatMatrixSparse:
    Clase generadora de formatos para matrices dispersas, tiene la capacidad de recibir matrices dentro de los formatos
    definidos ó generar una matriz dispersa aletoria de acuerdo al formato, dimensión y densidad definidos.
    '''

    def __init__(self, format, matrix=None, rows=3, cols=3, density=0.60):
        '''
           Método iniciador del objeto FormatMatrixSparse
           :param: format: fomato de la matriz, las opciones diposnibles son: coo, csr, csc, bsr
           :param: matrix: opcional matrix dispersa en alguno de los formatos
           :param: rows: opcional cantidad de filas para la matriz dispersa aleatoria
           :param: cols: opcional cantidad de columnas para la matriz dispersa aleatoria
        '''
        self.format = format
        self.format_matrix = matrix
        self.rows = rows
        self.cols = cols
        self.density = density
        try:
            FORMAT[self.format]
        except KeyError:
            print('FORMAT ERROR: {0} no es un formato valido'.format(self.format))
            raise
        if self.format_matrix is None:
            self.create_matrix()
        self.mtype()

    def create_matrix(self):
        '''
        Método generador de matrices aleatorias y formatos según las dimensiones y densidad definidas
        :return: format_matrix
        '''
        self.format_matrix = random(self.rows, self.cols, format=self.format, density=self.density)
        return self.format_matrix

    def get_by_rows(self):
        '''
        Método que retorna valores diferentes de cero ordenados por filas
        :return: list
        '''
        value= []
        if self.format == 'coo':
            value = list(zip(self.format_matrix.data, self.format_matrix.row, self.format_matrix.col))
            value = sorted(value , key=lambda tup: (tup[1], tup[2]))
            value = [item[0] for item in value]
        elif self.format == 'csr':
            value = self.format_matrix.data
        elif self.format == 'csc':
            value = list(zip(self.format_matrix.data, self.format_matrix.indices))
            value = sorted(value, key=lambda tup: (tup[1]))
            value = [round(x[0],8) for x in value]
        elif self.format == 'bsr':
            for index, item in enumerate(self.format_matrix.indptr):
                row_block = 0
                if index + 1 < len(self.format_matrix.indptr):
                    start = self.format_matrix.indptr[index]
                    end = self.format_matrix.indptr[index + 1]
                    blocks = self.format_matrix.data[start:end]
                    while row_block < self.format_matrix.blocksize[0]:
                        for block in blocks:
                            value.extend(block[row_block])
                        row_block += 1
        return value

    def get_by_cols(self):
        '''
        Método que retorna valores diferentes de cero ordenados por columnas
        :return: list
        '''
        value = []
        if self.format == 'coo':
            value = list(zip(self.format_matrix.data, self.format_matrix.row, self.format_matrix.col))
            value = sorted(value , key=lambda tup: (tup[2], tup[1]))
            value = [item[0] for item in value]
        elif self.format == 'csr':
            value = list(zip(self.format_matrix.data, self.format_matrix.indices))
            value = sorted(value, key=lambda tup: (tup[1]))
            value = [round(x[0], 8) for x in value]
        elif self.format == 'css':
            value = self.format_matrix.data
        elif self.format == 'bsr':
            order = list(zip(self.format_matrix.data, self.format_matrix.indices))
            order = sorted(order, key=lambda tup: (tup[1]))
            for item in set(self.format_matrix.indices):
                blocks = list(filter(lambda tup: tup[1] == item, order))
                blocks = [x[0] for x in blocks]
                col_block = 0
                while col_block < self.format_matrix.blocksize[0]:
                    for block in blocks:
                        value.extend(block[col_block])
                    col_block += 1
        return value

    def get_submatrix(self, dimension):
        '''
        Método que retorna una submatriz cuadrada de la matriz original apartir de los valores en el formato y
        seguna la dimension definidad, para el caso del formato bsr la dimensión indica el bloque que se desea extraer
        de la matriz dispersa original.
        :return: numpy matrix
        '''
        submatrix = []
        line =[]
        if self.format == 'coo':
            coordinates=[]
            value = list(zip(self.format_matrix.data, self.format_matrix.row, self.format_matrix.col))
            value = sorted(value, key=lambda tup: (tup[1], tup[2]))
            coor = [(x[1], x[2]) for x in value]
            x, y = coor[0]
            max_row = x+dimension
            max_col = y+dimension
            if max_row <= self.rows and max_col <= self.cols:
                for item in range(x, max_row):
                    col = y
                    dim = 0
                    while dim < dimension:
                        coordinates.append((item, col))
                        col +=1
                        dim += 1
                    for item in  coordinates:
                        if item in coor:
                            line.append(value[coor.index(item)][0])
                        else:
                            line.append(0)
                        if len(line) == dimension:
                            submatrix.append(line)
                            line = []
                            coordinates =[]
                return np.matrix(submatrix)
            else:
                raise ValueError('Nos posible extrar la Submatriz con la dimensión y matriz actual')

        elif self.format == 'csr' or self.format == 'csc':
            fila = []
            count = 1
            colum = [min(self.format_matrix.indices)]
            while count < dimension:
                col = colum[0] + count
                colum.append(col)
                count += 1
            contador = 0
            for index, item in enumerate(self.format_matrix.indices):
                if item in colum:
                    diferencia = colum.index(item) - len(fila)
                    print(diferencia)
                    if diferencia == 0:
                        fila.append(self.format_matrix.data[index])
                    elif diferencia > 0:
                        fila.extend([0] * diferencia)
                        fila.append(self.format_matrix.data[index])
                contador += 1
                if contador == dimension:
                    dif = dimension - len(fila)
                    if dif > 0:
                        fila.extend([0] * dif)
                if len(fila) == dimension:
                    submatrix.append(fila)
                    fila = []
                    contador = 0
                if len(submatrix) == dimension:
                    if self.format == 'csc':
                        csc = []
                        cont = 0
                        fila =[]
                        while cont < dimension:
                            for item in submatrix:
                                fila.append(item[cont])
                            csc.append(fila)
                            fila = []
                            cont += 1
                        submatrix = csc
                    return np.matrix(submatrix)
        else:
            return np.matrix(self.format_matrix.data[0])


    def get_matrix(self):
        '''
        Método para imprimir y obtener la matrix dispersa, no recomendable para grandes matrices
        :return: numpy matrix
        '''
        print('Matriz dispersa aleatoria {0} x {1}:\n'.format(self.rows, self.cols), self.format_matrix.toarray(),'\n')
        return np.matrix(self.format_matrix.toarray())

    def get_format(self):
        '''
        Método que retorna los valores del formato
        :return: tuple
        '''
        if self.format == 'coo':
            print('\nFormato {1}: {0} \n non-zero, row, col\n'.format(FORMAT[self.format], self.format))
            return self.format_matrix.data, self.format_matrix.row, self.format_matrix.col
        else:
            print('\nFormato {1}: {0} \n non-zero, index, indptr\n'.format(FORMAT[self.format], self.format))
            return self.format_matrix.data, self.format_matrix.indices, self.format_matrix.indptr

    def mtype(self):
        '''
        Método que imprime el tipo de formato del objeto
        :return:
        '''
        print('\n', self.format_matrix.__repr__())

    def size_matrix(self):
        '''
        Método que retorna el tamaño de la matriz dispersa en bytes
        :return: int
        '''
        return self.format_matrix.toarray().nbytes

    def size_format(self):
        '''
        Método que retorna el tamaño del formato en bytes
        :return: int
        '''
        if self.format == 'coo':
            return self.format_matrix.row.nbytes + self.format_matrix.data.nbytes
        else:
            return  self.format_matrix.data.nbytes + self.format_matrix.indptr.nbytes + \
                    self.format_matrix.indices.nbytes

    def ratio_compression(self):
        '''
        Método para obtener la tasa de compresión del formato vs la matriz dispersa
        :return:
        '''
        return  (1-self.size_format()/self.size_matrix())

    def get_density(self):
        '''
        Método para obtener la densidad de la matriz
        :return:
        '''
        return self.format_matrix.nnz/self.format_matrix.toarray().size
