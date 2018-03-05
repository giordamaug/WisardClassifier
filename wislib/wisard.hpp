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
    unsigned long int mypowers[64] = {
        1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL, 128UL, 256UL, 512UL, 1024UL, 2048UL, 4096UL, 8192UL, 16384UL, 32768UL, 65536UL, 131072UL , 262144UL, 524288UL,
        1048576UL, 2097152UL, 4194304UL, 8388608UL, 16777216UL, 33554432UL, 67108864UL, 134217728UL, 268435456UL, 536870912UL, 1073741824UL, 2147483648UL,
        4294967296UL, 8589934592UL, 17179869184UL, 34359738368UL, 68719476736UL, 137438953472UL, 274877906944UL, 549755813888UL, 1099511627776UL, 2199023255552UL, 4398046511104UL, 8796093022208UL, 17592186044416UL, 35184372088832UL, 70368744177664UL, 140737488355328UL, 281474976710656UL, 562949953421312UL, 1125899906842624UL, 2251799813685248UL, 4503599627370496UL, 9007199254740992UL, 18014398509481984UL, 36028797018963968UL, 72057594037927936UL, 144115188075855872UL, 288230376151711744UL, 576460752303423488UL, 1152921504606846976UL, 2305843009213693952UL, 4611686018427387904UL, 9223372036854775808UL
    };
    
    /// discriminator data structure
    /**
     represent a disciminator and its configuration and size info
     */
    typedef struct {
        int n_ram;          /**< number of rams for the discriminator */
        int n_bit;          /**< number of bits (resolution) */
        int n_loc;          /**< number of location in each ram (minus 1) */
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
