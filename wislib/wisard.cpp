//
//  wisard.cpp
//
//
//  Created by Maurizio Giordano on 20/03/2014
//
// the WISARD C++ implementation
//
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "wisard.hpp"
#include <iostream>
#include <stdexcept>

extern unsigned long int mypowers[];
using namespace std;

extern "C"  //Tells the compile to use C-linkage for the next scope.
{
    
    // mapping function
    int **mapping(int size, const char *mode, unsigned int seed) {
        register int i;
        int **maps;
        if (seed>0) srand (seed); else srand (time(NULL));
        maps = (int **)malloc(2 * sizeof(int *));
        int *list = (int *)malloc(size * sizeof(int));
        // revers mapping (not necessary yet!)
        int *rlist = (int *)malloc(size * sizeof(int));
        if (strcmp("linear", mode)==0) {
            for (i = 0; i < size; i++) {
                list[i] = i;
                rlist[i] = i;
            }
            maps[0] = (int *)list;
            maps[1] = (int *)rlist;
            return maps;
        } else if (strcmp("empty", mode)==0) {
            for (i = 0; i < size; i++) {
                list[i] = 0;
                rlist[i] = 0;
            }
            maps[0] = (int *)list;
            maps[1] = (int *)rlist;
            return maps;
        } else if (strcmp("random", mode)==0) {
            for (i = 0; i < size; i++) {
                list[i] = i;
                rlist[i] = i;
            }
            int *vektor = (int *)malloc(size * sizeof(int));
            for (i = 0; i < size; i++) {
                int j = i + rand() % (size - i);
                int temp = list[i];
                list[i] = list[j];
                rlist[list[j]] = i;
                list[j] = temp;
                vektor[i] = list[i];
            }
            maps[0] = (int *)vektor;
            maps[1] = (int *)rlist;
            return maps;
        } else {
            throw std::invalid_argument( "received wrong mapping mode" );
        }
    }
    //! DISCRIMINATOR CONSTRUCTOR
    /*!
     \param n_bit number of bits (resolution)
     \param size input binary image size
     \param name discriminator name
     \param mode retina creation mode (linear, random)
     \return a pointer to the discriminator data structure
     */
    discr_t *make_discr(int n_bit, int size, const char *mode, unsigned int seed) {
        int i,j;
        discr_t *p = (discr_t *)malloc(sizeof(discr_t));
        int **maps;
        p->n_bit = n_bit;
        p->n_loc = mypowers[n_bit];
        p->size = size;
        p->tcounter = 0;
        p->mi = (float *)malloc(size * sizeof(float));
        for (i=0; i<size;i++) { p->mi[i] = (float) 0; }
        if (size % n_bit == 0) p->n_ram = (int)(size / n_bit);  // set no. of rams
        else p->n_ram = (int)(size / n_bit) + 1;
        maps = (int **)mapping(size, mode ,seed);      // set mapping (linear or random)
        p->map = maps[0];
        p->rmap = maps[1];
        p->rams = (float **)malloc(p->n_ram * sizeof(float *));
        for (i=0;i<p->n_ram;i++) {                      // alloc rams and zeroed
            p->rams[i] = (float *)malloc(p->n_loc * sizeof(float));
            for (j=0;j<p->n_loc;j++) p->rams[i][j] = (float)0;
        }
        return p;
    }
    
    //! DISCRIMINATOR TRAINING FUNCTION
    /*! train_discr
     \param discr       - pointer to the discriminator to be trained
     \param in_tuples   - input tuple for training the discriminator
     \return void
     */
    void train_discr(discr_t *discr, unsigned int *in_tuples) {
        int neuron;
        discr->tcounter++;
        for (neuron=0;neuron<discr->n_ram;neuron++)
            discr->rams[neuron][in_tuples[neuron]] += (float)1;
    }
    /*! train_tuple
     \param rams        - array of rams to be trained
     \param n_ram       - no. of rams
     \param in_tuples   - input tuple for training rams
     \return void
     */
    void train_tuple(float **rams, int n_ram, unsigned int *in_tuples) {
        int neuron;
        for (neuron=0;neuron<n_ram;neuron++) {
            rams[neuron][in_tuples[neuron]] += (float)1;
        }
    }
    /*! train_libsvm
     \param rams        - array of rams to be trained
     \param map         - input-to-ram mapping
     \param n_ram       - no. of rams
     \param n_bit       - no. of bits
     \param n_tics      - scaling range
     \param n_feature   - no. of variables in datum
     \param data        - multivariable datum (real vector)
     \param den         - variable ranges (real vector)
     \param off         - variable offset (real vector)
     \return void
     */
    void train_libsvm(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data, double *den, double *off) {
        int neuron, i;
        int address;
        int x, index, npixels=n_tics * n_feature, value;
        for (neuron=0;neuron<n_ram;neuron++) {
            // compute neuron simulus
            address=(unsigned int)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bit;i++) {
                x = map[((neuron * n_bit) + i) % npixels];
                index = x/n_tics;
                value = (int) ((data[index] - off[index]) * n_tics / den[index]);
                if ((x % n_tics) < value) {
                    address |= mypowers[n_bit -1 - i];
                }
            }
            rams[neuron][address] += 1.0;
        }
    }
    /*! train_libsvm_noscale
     \param rams        - array of rams to be trained
     \param map         - input-to-ram mapping
     \param n_ram       - no. of rams
     \param n_bit       - no. of bits
     \param n_tics      - scaling range
     \param n_feature   - no. of variables in datum
     \param data        - multivariable datum (real vector)
     \return void
     */
    void train_libsvm_noscale(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data) {
        int neuron, i;
        int address;
        int x, index, npixels=n_tics * n_feature, value;
        for (neuron=0;neuron<n_ram;neuron++) {
            address=(unsigned int)0;            // compute neuron simulus
            // decompose record data values into wisard input
            for (i=0;i<n_bit;i++) {
                x = map[((neuron * n_bit) + i) % npixels];
                index = x/n_tics;
                value = (int) (data[index] * n_tics);
                if ((x % n_tics) < value) {
                    address |= mypowers[n_bit -1 - i];
                }
            }
            rams[neuron][address] += 1.0;
        }
    }
    
    // CLASSIFYING FUNCTIONS
    /*! classify_discr
     \param discr       - pointer to the discriminator thta classifies
     \param in_tuples   - input tuple for training the discriminator
     \return response of rams (range [0,1])
     */
    double classify_discr(discr_t *discr, unsigned int *in_tuples) {
        int neuron, sum;
        
        for (neuron=0, sum=0;neuron<discr->n_ram;neuron++)
            if (discr->rams[neuron][in_tuples[neuron]] > 0) sum++;
        return (double)sum/(double)discr->n_ram;
    }
    /*! response_discr
     \param discr       - pointer to the discriminator thta classifies
     \param in_tuples   - input tuple for training the discriminator
     \return array of ram outputs (reals)
     */
    double *response_discr(discr_t *discr, unsigned int *in_tuples) {
        int neuron;
        double *res = (double *)malloc(discr->n_ram * sizeof(double));
        
        for (neuron=0;neuron<discr->n_ram;neuron++)
            res[neuron] = discr->rams[neuron][in_tuples[neuron]];
        return res;
    }
    /*! classify_tuple
     \param rams        - array of rams that classifies
     \param n_ram       - no. of rams
     \param in_tuples   - input tuple for training rams
     \return response of rams (range [0,1])
     */
    double classify_tuple(float **rams, int n_ram, unsigned int *in_tuples) {
        int neuron, sum;
        
        for (neuron=0, sum=0;neuron<n_ram;neuron++)
            if (rams[neuron][in_tuples[neuron]] > 0) sum++;
        return (double)sum/(double)n_ram;
    }
    /*! response_tuple
     \param rams        - array of rams that classifies
     \param n_ram       - no. of rams
     \param in_tuples   - input tuple for training rams
     \return array of ram outputs (reals)
     */
    float *response_tuple(float **rams, int n_ram, unsigned int *in_tuples) {
        int neuron;
        float *res = (float *)malloc(n_ram * sizeof(float));

        for (neuron=0;neuron<n_ram;neuron++)
            res[neuron] = rams[neuron][in_tuples[neuron]];
        return res;
    }
    /*! classify_libsvm
     \param rams        - array of rams that classifies
     \param map         - input-to-ram mapping
     \param n_ram       - no. of rams
     \param n_bit       - no. of bits
     \param n_tics      - scaling range
     \param n_feature   - no. of variables in datum
     \param data        - multivariable datum (real vector)
     \param den         - variable ranges (real vector)
     \param off         - variable offset (real vector)
     \return response of rams (range [0,1])
     */
    double classify_libsvm(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data, double *den, double *off) {
        int neuron, sum=0;
        unsigned int address;
        int x, i, index, npixels=n_tics * n_feature, value;
        
        //#pragma omp parallel for schedule(static) shared(discr) private(neuron,address)
        for (neuron=0;neuron<n_ram;neuron++) {
            // compute neuron simulus
            address=(unsigned int)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bit;i++) {
                x = map[((neuron * n_bit) + i) % npixels];
                index = x/n_tics;
                value = (int) ((data[index] - off[index]) * n_tics / den[index]);
                if ((x % n_tics) < value) {
                    address |= (unsigned int)mypowers[n_bit -1 - i];
                }
            }
            if (rams[neuron][address] > 0) {
                sum++;
            }
        }
        return (double)sum/(double)n_ram;
    }
    /*! response_libsvm
     \param rams        - array of rams that classifies
     \param map         - input-to-ram mapping
     \param n_ram       - no. of rams
     \param n_bit       - no. of bits
     \param n_tics      - scaling range
     \param n_feature   - no. of variables in datum
     \param data        - multivariable datum (real vector)
     \param den         - variable ranges (real vector)
     \param off         - variable offset (real vector)
     \return array of ram outputs (reals)
     */
    double *response_libsvm(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data, double *den, double *off) {
        int neuron;
        unsigned int address;
        int x, i, index, npixels=n_tics * n_feature, value;
        double *res = (double *)malloc(n_ram * sizeof(double));
        
        for (neuron=0;neuron<n_ram;neuron++) {
            address=(unsigned int)0;            // compute neuron simulus
            // decompose record data values into wisard input
            for (i=0;i<n_bit;i++) {
                x = map[((neuron * n_bit) + i) % npixels];
                index = x/n_tics;
                value = (int) ((data[index] - off[index]) * n_tics / den[index]);
                if ((x % n_tics) < value) {
                    address |= (unsigned int)mypowers[n_bit -1 - i];
                }
            }
            res[neuron] = rams[neuron][address];
        }
        return res;
    }
    /*! classify_libsvm_noscale
     \param rams        - array of rams that classifies
     \param map         - input-to-ram mapping
     \param n_ram       - no. of rams
     \param n_bit       - no. of bits
     \param n_tics      - scaling range
     \param n_feature   - no. of variables in datum
     \param data        - multivariable datum (real vector)
     \return response of rams (range [0,1])
     */
    double classify_libsvm_noscale(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data) {
        int neuron, sum=0;
        unsigned int address;
        int x, i, index, npixels=n_tics * n_feature, value;
        
        //#pragma omp parallel for schedule(static) shared(discr) private(neuron,address)
        for (neuron=0;neuron<n_ram;neuron++) {
            // compute neuron simulus
            address=(unsigned int)0;
            // decompose record data values into wisard input
            for (i=0;i<n_bit;i++) {
                x = map[((neuron * n_bit) + i) % npixels];
                index = x/n_tics;
                value = (int) (data[index] * n_tics);
                if ((x % n_tics) < value) {
                    address |= (unsigned int)mypowers[n_bit -1 - i];
                }
            }
            if (rams[neuron][address] > 0) {
                sum++;
            }
        }
        // store responses
        return (double)sum/(double)n_ram;
    }
    /*! response_libsvm_noscale
     \param rams        - array of rams that classifies
     \param map         - input-to-ram mapping
     \param n_ram       - no. of rams
     \param n_bit       - no. of bits
     \param n_tics      - scaling range
     \param n_feature   - no. of variables in datum
     \param data        - multivariable datum (real vector)
     \return array of ram outputs (reals)
     */
    double *response_libsvm_noscale(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data) {
        int neuron, sum=0;
        unsigned int address;
        int x, i, index, npixels=n_tics * n_feature, value;
        double *res = (double *)malloc(n_ram * sizeof(double));
        
        for (neuron=0;neuron<n_ram;neuron++) {
            address=(unsigned int)0;            // compute neuron simulus
            // decompose record data values into wisard input
            for (i=0;i<n_bit;i++) {
                x = map[((neuron * n_bit) + i) % npixels];
                index = x/n_tics;
                value = (int) (data[index] * n_tics);
                if ((x % n_tics) < value) {
                    address |= (unsigned int)mypowers[n_bit -1 - i];
                }
            }
            res[neuron] = rams[neuron][address];
        }
        return res;
    }
    /*! UTILITY FUNCTION
     \param discr       - pointer to the discriminator thta classifies
     \return void
    */
     void print_discr(discr_t *discr) {
         int neuron, j;
         fprintf(stdout,"[n.bits: %d\n size: %d\n n.rams: %d\n n.loc: %d\n rams:",discr->n_bit,discr->size,discr->n_ram,discr->n_loc);
         fprintf(stdout,"{");
         for (j=0;j<discr->n_loc;j++)
             if (discr->rams[0][j] > 0) fprintf(stdout,"%d:%f ", j, discr->rams[0][j]);
         fprintf(stdout,"}\n");
         for (neuron=1;neuron<discr->n_ram;neuron++) {
            fprintf(stdout,"      {");
            for (j=0;j<discr->n_loc;j++)
                if (discr->rams[neuron][j] > 0) fprintf(stdout,"%d:%f ", j, discr->rams[neuron][j]);
            fprintf(stdout,"}\n");
         }
         fprintf(stdout," map: [");
         for (j=0; j < discr->size; j++) printf("%d ", discr->map[j]);
         fprintf(stdout,"]\n");
         fprintf(stdout,"]\n");
     }

}
