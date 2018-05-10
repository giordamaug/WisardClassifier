//
//  wisard.hpp
//
//
//  Created by Maurizio Giordano on 20/03/2014
//
//
#ifndef _wisard_h
#define _wisard_h

#define PI 3.1415926535

#include <assert.h>
#include <stdio.h>


extern "C"  //Tells the compile to use C-linkage for the next scope.
{
    
#define BUFSIZE 1024
#define ceiling(X) (X-(int)(X) > 0 ? (int)(X+1) : (int)(X))
    unsigned int mypowers[32] = {
        1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U, 1024U, 2048U, 4096U, 8192U, 16384U, 32768U, 65536U, 131072U , 262144U, 524288U,
        1048576U, 2097152U, 4194304U, 8388608U, 16777216U, 33554432U, 67108864U, 134217728U, 268435456U, 536870912U, 1073741824U, 2147483648U };
    
    /// discriminator data structure
    /**
     represent a disciminator and its configuration and size info
     */
    typedef struct {
        int n_ram;          /**< number of rams for the discriminator */
        int n_bit;          /**< number of bits (resolution) */
        unsigned int n_loc;          /**< number of location in each ram (minus 1) */
        int size;           /**< size of input binary image */
        unsigned long int tcounter;  /**< train counter */
        float **rams;       /**< the ram list of disciminator */
        int *map;           /**< pointer to the retina (mapping) */
        int *rmap;          /**< pointer to the inverse retina (mapping) */
        float *mi;          /**< pointer to mental image */
    } discr_t;
    //! DISCRIMINATOR CONSTRUCTOR.
    discr_t *make_discr(int n_bit, int size, const char *mode, unsigned int seed);
    void print_discr(discr_t *discr);
    //! DISCRIMINATOR TRAINING FUNCTION
    void train_discr(discr_t *discr, unsigned int *in_tuples);
    void train_tuple(float **rams, int n_ram, unsigned int *in_tuples);
    void train_libsvm(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data, double *den, double *off);
    void train_libsvm_noscale(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data);
    
    // CLASSIFYING FUNCTIONS
    double classify_discr(discr_t *discr, unsigned int *in_tuples);
    double *response_discr(discr_t *discr, unsigned int *in_tuples);
    double classify_tuple(float **rams, int n_ram, unsigned int *in_tuples);
    float *response_tuple(float **rams, int n_ram, unsigned int *in_tuples);
    double classify_libsvm(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data, double *den, double *off);
    double *response_libsvm(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data, double *den, double *off);
    double classify_libsvm_noscale(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data);
    double *response_libsvm_noscale(float **rams, unsigned int *map, int n_ram, int n_bit, int n_tics, int n_feature, double *data);
}
#endif
