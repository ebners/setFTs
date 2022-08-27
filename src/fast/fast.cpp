#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
using namespace std;
namespace py = pybind11;

//fast discrete set fourier transform 3 (self-inverse version)
vector<double> fdsft3_selfInverse(vector<double> signal){
    int len = signal.size();
    //check if len is power of two
    if((len != 0) && (len & (len - 1))){
        throw std::invalid_argument("Length of signal must be power of two");
    }
    int h = 1;

    vector<double> transform;
    transform = signal;

    while(h < len){
        for(int i = 0; i < len; i += 2*h){
            for(int j = i;j < i + h; j++){
                double x = transform[j];
                double y = transform[j+h];
                transform[j] = x;
                transform[j+h] = x - y;
            }
        }
        h = h*2;
    }
    return transform;
}

//fast discrete set fourier transform 3 (positives only)
vector<double> fdsft3(vector<double> signal){
    int len = signal.size();
    if((len != 0) && (len & (len - 1))){
        throw std::invalid_argument("Length of signal must be power of two");
    }
    int h = 1;

    vector<double> transform;
    transform = signal;

    while(h < len){
        for(int i = 0; i < len; i += 2*h){
            for(int j = i;j < i + h; j++){
                double x = transform[j];
                double y = transform[j+h];
                transform[j] = x;
                transform[j+h] = -x + y;
            }
        }
        h = h*2;
    }
    return transform;
}

// inverse of fast discrete set fourier transform 3 (positives only)
vector<double> fidsft3(vector<double> signal){
    int len = signal.size();
    if((len != 0) && (len & (len - 1))){
        throw std::invalid_argument("Length of signal must be power of two");
    }
    int h = 1;

    vector<double> transform;
    transform = signal;

    while(h < len){
        for(int i = 0; i < len; i += 2*h){
            for(int j = i;j < i + h; j++){
                double x = transform[j];
                double y = transform[j+h];
                transform[j] = x;
                transform[j+h] = x + y;
            }
        }
        h = h*2;
    }
    return transform;
}

//fast discrete set fourier transform 4
vector<double> fdsft4(vector<double> signal){
    int len = signal.size();
    if((len != 0) && (len & (len - 1))){
        throw std::invalid_argument("Length of signal must be power of two");
    }
    int h = 1;

    vector<double> transform;
    transform = signal;

    while(h < len){
        for(int i = 0; i < len; i += 2*h){
            for(int j = i;j < i + h; j++){
                double x = transform[j];
                double y = transform[j+h];
                transform[j] = y;
                transform[j+h] = x - y;
            }
        }
        h = h*2;
    }
    return transform;
}
//inverse of fast discrete set fourier transform 4
vector<double> fidsft4(vector<double> signal){
    int len = signal.size();
    if((len != 0) && (len & (len - 1))){
        throw std::invalid_argument("Length of signal must be power of two");
    }
    int h = 1;

    vector<double> transform;
    transform = signal;

    while(h < len){
        for(int i = 0; i < len; i += 2*h){
            for(int j = i;j < i + h; j++){
                double x = transform[j];
                double y = transform[j+h];
                transform[j] = x + y;
                transform[j+h] = x;
            }
        }
        h = h*2;
    }
    return transform;
}

//fast walsh-hadamard Transform
vector<double> fwht(vector<double> signal){
    int len = signal.size();
    if((len != 0) && (len & (len - 1))){
        throw std::invalid_argument("Length of signal must be power of two");
    }
    int h = 1;

    vector<double> transform;
    transform = signal;

    while(h < len){
        for(int i = 0; i < len; i += 2*h){
            for(int j = i;j < i + h; j++){
                double x = transform[j];
                double y = transform[j+h];
                transform[j] =(x + y)/2;
                transform[j+h] = (x - y)/2;
            }
        }
        h = h*2;
    }
    //normalize
    // for(int i = 0; i < len; i++){
    //     transform[i] = transform[i]/(pow(2,len));
    // }

    return transform;

}

// inverse of fast walsh-hadamard Transform
vector<double> fiwht(vector<double> signal){
    int len = signal.size();
    if((len != 0) && (len & (len - 1))){
        throw std::invalid_argument("Length of signal must be power of two");
    }
    int h = 1;

    vector<double> transform;
    transform = signal;

    while(h < len){
        for(int i = 0; i < len; i += 2*h){
            for(int j = i;j < i + h; j++){
                double x = transform[j];
                double y = transform[j+h];
                transform[j] =x + y;
                transform[j+h] = x - y;
            }
        }
        h = h*2;
    }
    return transform;
}




PYBIND11_MODULE(fast, m) {
    m.doc() = "Fast Discrete Set Fourier Transform";
    m.def("fdsft3", &fdsft3, "Fast Discrete Set Fourier Transform 3");
    m.def("fdsft3_selfInverse", &fdsft3_selfInverse, "Fast Discrete Set Fourier Transform 3 (self-inverse version)");
    m.def("fdsft4", &fdsft4, "Fast Discrete Set Fourier Transform 4");
    m.def("fidsft3", &fidsft3, "Inverse of Fast Discrete Set Fourier Transform 3");
    m.def("fidsft4", &fidsft4, "Inverse of Fast Discrete Set Fourier Transform 4");
    m.def("fwht", &fwht, "Fast Walsh-Hadamard Transform");
    m.def("fiwht", &fiwht, "Inverse of Fast Walsh-Hadamard Transform");
}
