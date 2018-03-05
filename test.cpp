//
//  test.cpp
//
//
//  Created by Maurizio Giordano on 20/03/2014
//
// the WISARD C++ implementation
//

#include "wisard.hpp"
#include <iostream>
#include <string>

unsigned int *mk_tuple(discr_t *discr, int *sample) {
    int i,j;
    /* alloc tuple array */
    unsigned int *intuple = (unsigned int *)malloc(discr->n_ram * sizeof(unsigned int));
    int x;
    for (i = 0; i < discr->n_ram; i++)
        for (j = 0; j < discr->n_bit; j++) {
            x = discr->map[(i * discr->n_bit) + j] % discr->size;
            intuple[i] += (1<<(discr->n_bit -1 - j))  * sample[x];
    }
    return intuple;
}

int main() {
   
    int X[8][8] ={{0, 1, 0, 0, 0, 0, 0, 0},
                  {0, 0, 1, 1, 1, 1, 0, 0},
                  {0, 0, 1, 0, 0, 0, 1, 0},
                  {1, 0, 0, 0, 0, 0, 0, 1},
                  {1, 1, 0, 1, 1, 1, 1, 1},
                  {1, 0, 0, 0, 0, 0, 0, 0},
                  {0, 0, 0, 0, 1, 0, 0, 1},
                  {1, 0, 0, 0, 0, 0, 0, 1}};
    std::string y[8] = {"A","A","B","B","A","A","B","A"};
    double responses[2];
    int s;
    int test[8] = {0, 0, 1, 0, 0, 0, 1, 0};
    
    // init WiSARD (create discriminator for each class "A" and "B")
    discr_t wisard[2];
    wisard[0] = *make_discr(2,8,"random",0);
    wisard[1] = *make_discr(2,8,"random",0);

    // train WiSARD
    for (s=0; s < 8; s++)
        if (y[s] == "A")
            train_discr(wisard,mk_tuple(wisard,X[s]));
        else
            train_discr(wisard+1,mk_tuple(wisard+1,X[s]));
    
    // print WiSARD state
    print_discr(wisard);
    print_discr(wisard+1);
    
    // predict by WiSARD
    responses[0] = classify_tuple(wisard->rams,wisard->n_ram,mk_tuple(wisard,test));
    responses[1] = classify_tuple((wisard+1)->rams,(wisard+1)->n_ram,mk_tuple(wisard+1,test));
    printf("A responds with score %.2f\n",responses[0]);
    printf("B responds with score %.2f\n",responses[1]);
}
