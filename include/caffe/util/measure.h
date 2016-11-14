#ifndef MEASURE_H
#define MEASURE_H
#include <vector>
class measure
{
public:
	measure();
	~measure();

public:
	float m_nmi(int* label_anchor, int* label_predict, int length);
	float m_nmi_fast(int* label_anchor, int* label_predict, int length);
};
#endif
