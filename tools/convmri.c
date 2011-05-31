
#include <stdio.h>
#include <stdint.h>


void usage(const char* name) {
    fprintf(stderr,"usage: %s <input> <output>\n",name);
}

int main(int argc, char* argv[]) {

    if (argc != 3)  {
        usage(argv[0]);
        return 1;
    }
    
    FILE *input = fopen(argv[1], "r");
    if (!input) {
        fprintf(stderr, "Could not open input file: %s\n",argv[1]);
        return 2;
    }
    FILE *output = fopen(argv[2], "w");
    if (!output) {
        fprintf(stderr, "Counld not open output file: %s\n",argv[2]);
        return 3;
    }
    
    
    float min = 1000000.0;
    float max = -1000000.0;
    
    float f;
    while (fread(&f,sizeof(float),1,input)) {
        if (f < min)
            min = f;
        if (f > max)
            max = f;
    }
    
    printf("min = %f, max = %f\n",min,max);
    
    float span = max-min;
    
    fseek(input, 0, SEEK_SET);
    while (fread(&f,sizeof(float),1,input)) {
        f = (f - min)/span;
        
        uint16_t u16 = f * UINT16_MAX;
        fwrite(&u16,sizeof(uint16_t),1,output);
    }
    
    
    fclose(input);
    fclose(output);
    

    return 0;
}
