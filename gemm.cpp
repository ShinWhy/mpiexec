#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
//command line sample: mpirun -n 1 ./gemm 1024
/** 这个命令是用于在MPI（Message Passing Interface）环境中运行程序的。其中：
  * "mpirun" 是用来启动 MPI 应用程序的命令。
  * "-n 1" 指定要启动的进程数量，这里是1个进程。
  * "./gemm" 是要运行的可执行文件。
  * "1024" 是传递给程序的参数，可能是矩阵乘法的维度大小。1024*1024
*/
int main(int argc, char *argv[]) 
{ //argc and argv  is MPI-related command line arguments
    double start, stop;
    int i, j, k, l;
    int *a, *b, *c, *buffer, *ans; 
    int size = atoi(argv[1]);//用于将字符串转换为整数，矩阵的维度大小
    int rank, numprocs, line;

    MPI_Init(&argc, &argv); //MPI initialization
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //获得当前进程号，使用 MPI_COMM_WORLD 来进行全局的通信操作，使得所有进程都能够相互通信和交换数据。
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);//获得进程个数

    line = size/numprocs; //将数据分成‘进程数’个块，主进程也要处理数据
    a = (int*)malloc(sizeof(int)*size*size);
    b = (int*)malloc(sizeof(int)*size*size);
    c = (int*)malloc(sizeof(int)*size*size);
    //缓存大于等于要处理的数据大小，大于时只需要关注实际数据那部分
    buffer = (int*)malloc(sizeof(int)*size*line);//数据分组大小
    ans = (int*)malloc(sizeof(int)*size*line);//保存数据块计算的结果

    //主进程对矩阵M赋初值，并将矩阵N广播到个进程，将矩阵M分组广播到个进程
    if(rank == 0){
      printf("task %d start\n)", rank);
      //从文件中读入矩阵
      FILE *fp;

      fp = fopen("a.txt", "r");
      start = MPI_Wtime();
      for(i = 0; i < size; i++){
          for(j = 0; j < size; j++){
            a[size * i + j] = size * i + j;
          }
      }
      //将矩阵N发送给其他从进程
      for(i = 1; i < numprocs; i++){
        MPI_Send(b, size * size, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
      //将a的各行依次发给其他从进程
      for(l = 1; l < numprocs; l++){
        MPI_Send(a + (l - 1) * line * size, size * line, MPI_INT, l, 1, MPI_COMM_WORLD);
      }
      //接收从进程计算的结果
      for(k = 1; k < numprocs; k++) {
        MPI_Recv(ans, line*size, MPI_INT, k, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //将结果传给数组c
        for(i = 0; i < line; i++){
          for(j = 0; j < size; j++) {
            c[((k-1)*line+i)*size+j] = ans[i*size+j];
          }

        }
      }
      //计算a剩下的数据
      for(i = (numprocs-1)*line; i < size; i++){
        for(j = 0; j < size; j++){
          int temp = 0;
          for(k = 0; k < size; k++)
            temp += a[i*size+k] * b[k*size+j];
          c[i*size+j] = temp;
        }
      }
      fp = fopen("c.txt", "w");
      for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
          fprintf(fp, "%d", c[i*size+j]);
        }
        fputc('\n', fp);
      }
      fclose(fp);
      //结果测试
      //统计时间
      stop = MPI_Wtime();
      printf("task %d end\n", rank);
      printf("rank: %d time: %lfs\n", rank, stop-start);

      free(a);
      free(b);
      free(c);
      free(buffer);
      free(ans);
    }


    //其他进程接收数据，计算结果后，发送给主进程
    else{
      printf("task %d start\n", rank);
      //接收广播的数据（矩阵b)
      MPI_Recv(b, size*size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Recv(buffer, size*line, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //计算乘积结果，并将结果发送给主进程
      for(i = 0; i < line; i++){
        for(j = 0; j < size; j++){
          int temp = 0;
          for(k = 0; k < size; k++)
            temp += buffer[i*size + k] + b[k*size + j];
          ans[i * size + j] = temp;
        }
      }
      MPI_Send(ans, line*size, MPI_INT, 0, 3, MPI_COMM_WORLD);
      printf("task %d end\n", rank);

    }

    MPI_Finalize();

    return 0;




    

}