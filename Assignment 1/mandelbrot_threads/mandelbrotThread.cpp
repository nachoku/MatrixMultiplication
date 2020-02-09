#include <stdio.h>
#include <pthread.h>

#include "CycleTimer.h"

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
    int parts;
} WorkerArgs;


extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);


//
// workerThreadStart --
//
// Thread entrypoint.
void* workerThreadStart(void* threadArgs) {

    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);

    // TODO: Implement worker thread here.
    // Timing element added
	double startTime = CycleTimer::currentSeconds();
    	//Divide the execution into 2 calls- first from top half of image, second from corresponding second half of image
	//Num of rows is half of thread total for each call
	//Manipulate start row for second call by adding half the height
	mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, (args->parts) * (args->threadId)/2,args->parts/2, args->maxIterations, args->output);
	mandelbrotSerial(args->x0, args->y0, args->x1, args->y1, args->width, args->height, (args->parts) * (args->threadId)/2+args->height/2, args->parts/2, args->maxIterations, args->output);
	double endTime = CycleTimer::currentSeconds();
	double minThread = 1e30;
	minThread =  endTime - startTime;
	printf("Thread %d (%d): %.3f \n", args->threadId, args->parts, minThread );
    return NULL;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }
    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];
    //execution set for each thread is height divided by the number of threads
    int parts_per_thread= height/numThreads;
    for (int i=0; i<numThreads; i++) {
        // TODO: Set thread arguments here.
        // Added arguments to pass to threads
        args[i].threadId = i;
	args[i].height = height;
	args[i].width = width;
	args[i].x0 = x0;
	args[i].y0=y0;
	args[i].x1=x1;
	args[i].y1=y1;
	args[i].maxIterations=maxIterations;
	args[i].output=output;
	args[i].parts=parts_per_thread;
    }

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.

    for (int i=1; i<numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i=0; i<numThreads; i++)
	{
        pthread_join(workers[i], NULL);
	}
    printf("\n ---\n");
}
