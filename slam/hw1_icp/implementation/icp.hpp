class Myicp
{
private:
    int m_iterations;
public:
    Myicp(int iterations);
    ~Myicp();
};

Myicp::Myicp(int iterations): m_iterations(iterations)
{}

Myicp::~Myicp()
{}
