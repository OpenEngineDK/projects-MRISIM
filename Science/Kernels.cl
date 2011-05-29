

typedef struct PhantomInfo {
    int4 offset;
    float4 size;
} PhantomInfo;

typedef struct SpinPack {
    float t1;
    float t2;        
} SpinPack;

#define GYRO_HERTZ 42.576*1e6 // herz/tesla
#define GYRO_RAD GYRO_HERTZ * 2.0 * 3.14159 // (radians/s)/tesla

float4 RotateZ(float angle, float4 vec) {
     float4 v;
     v.x = vec.x*cos(angle) + vec.y*-sin(angle);
     v.y = vec.x*sin(angle) + vec.y*cos(angle);
     v.z = vec.z;

     return v;
}

float4 RotateX(float angle, float4 vec) {
     float4 v;
     v.x = vec.x;
     v.y = vec.y*cos(angle) + vec.z*sin(angle);
     v.z = vec.y*-sin(angle) + vec.z*cos(angle);

     return v;
}

float4 RotateY(float angle, float4 vec) {
     float4 v;
     v.x = vec.x*cos(angle) + vec.z*sin(angle);
     v.y = vec.y;
     v.z = vec.x*-sin(angle) + vec.z*cos(angle);

     return v;
}

__kernel void mri_step(float dt,                     // 0
                       __global float* refM,         // 1
                       __global unsigned char* data, // 2
                       __global SpinPack* packs,     // 3 
                       __global float* eq,           // 4
                       float4 gradient4,             // 5
                       float4 phantomOffset,         // 6
                       float4 phantomSize,           // 7
                       float2 rf                     // 8
                       ) {           


    int width = get_global_size(0);
    int height = get_global_size(1);
    int depth = get_global_size(2);
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int i = x + y*width + z*width*height;
    int vi = x*3 + y*width*3 + z*width*height*3;

    int type = data[i];


    float4 refMagnet;
    refMagnet.x = refM[vi+0];
    refMagnet.y = refM[vi+1];
    refMagnet.z = refM[vi+2];


    if (type != 0) {
        float dtt1 = dt/packs[type].t1;
        float dtt2 = dt/packs[type].t2;


        //refMagnet += (float4)(-refMagnet.x*dtt2,
        //                      -refMagnet.y*dtt2,
        //                      (eq[i]-refMagnet.z)*dtt1,
        //                      0);


        refMagnet = RotateX(rf.x * GYRO_RAD * dt, refMagnet);
        refMagnet = RotateY(rf.y * GYRO_RAD * dt, refMagnet);


	float e1 = exp(-dtt1);
	float e2 = exp(-dtt2);
        refMagnet = (float4)(e2 * refMagnet.x,
                             e2 * refMagnet.y,
                             e1 * refMagnet.z + eq[i] * (1.0 - e1),
                             0);

        // refMagnet = (float4)(refMagnet.x - (refMagnet.x * dt) / packs[type].t2,
	//            	     refMagnet.y - (refMagnet.y * dt) / packs[type].t2,
	// 		     refMagnet.z - (refMagnet.z - eq[i]) * dt / (packs[type].t1),
        //                      0);

        
        float4 v = (float4)((x + phantomOffset.x) * (phantomSize.x*1e-3),
                            (y + phantomOffset.y) * (phantomSize.y*1e-3),
                            (z + phantomOffset.z) * (phantomSize.z*1e-3),
                            0.0);
        

        float dG = dot(gradient4, v);
        
        float deltaB0 = 0.0;

        refMagnet = RotateZ(GYRO_RAD * (deltaB0 + dG) * dt, refMagnet);


    } else {
        refMagnet = (float4)(0.0, 0.0, 0.0, 0.0);
    }
    refM[vi+0] = refMagnet.x;
    refM[vi+1] = refMagnet.y;
    refM[vi+2] = refMagnet.z;
     
}

__kernel void reduce_signal(__global float* input, __global float* output) {

    int size = get_global_size(0);
    int x = get_global_id(0);
    
    
    int idx = x*2;
    
    int vi = idx*3;
    int vi2 = (idx+1)*3;
    
    /* float4 refMagnet; */
    /* refMagnet.x = input[vi+0]; */
    /* refMagnet.y = input[vi+1]; */
    /* refMagnet.z = input[vi+2]; */

    /* float4 refMagnet2; */
    /* refMagnet2.x = input[vi2+0]; */
    /* refMagnet2.y = input[vi2+1]; */
    /* refMagnet2.z = input[vi2+2]; */

    int ov = x*3;
    output[ov+0] = input[vi+0] + input[vi2+0];
    output[ov+1] = input[vi+1] + input[vi2+1];
    output[ov+2] = input[vi+2] + input[vi2+2];

   
}


