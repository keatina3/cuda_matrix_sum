#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>

int main(int argc, char **argv){
	int m = 10, n = 10;
	int t = 0, r = 0, option = 0;
	struct timeval seed;
	int i;

	while((option=getopt(argc,argv,"n:m:rt"))!=-1){
		switch(option){
			case 'n': n = atoi(optarg);
				break;
			case 'm': m = atoi(optarg);
				break;
			case 'r': r = 1;
				break;
			case 't': t = 1;
				break;
			default:
				printf("Incorrect options entered!\n");
				return 1;
		}
	}	
	if(argc != optind){
		printf("Too many arguments provided, exiting!\n");
		return 1;
	}

	gettimeofday(&seed, NULL);
	if(r)
		srand48(seed.tv_usec);
	else
		srand48(123456);
	
	printf("n = %d, m = %d, r = %d, t = %d, argc = %d\n",n,m,r,t,argc);
	
	return 0;
}
