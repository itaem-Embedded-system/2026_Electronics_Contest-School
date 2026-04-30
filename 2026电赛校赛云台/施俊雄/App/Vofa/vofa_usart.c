#include "stm32f1xx_hal.h"
#include "bsp_usart.h"   

uint8_t Usart_RxData;
uint8_t Usart_RxFlag;

uint8_t Usart_RxPacket[100];
uint8_t Usart_RxPacket_Len=100;

uint8_t Usart_GetRxFlag(void) 
{
	if(Usart_RxFlag==1)
	{
		Usart_RxFlag=0;
		return 1;
	}
	return 0;
}

uint8_t Usart_GetRxData(void)
{
	return Usart_RxData;
}

uint8_t id_Flag;
uint8_t Data_BitNum=0;

static uint8_t RxState=0;
static uint8_t pRxPacket=0;

void vofa_process_byte(uint8_t data)
{
	Usart_RxData = data;
	
	if(RxState==0&&Usart_RxData==0x23) 
	{
		RxState=1;
	}
	else if(RxState==1&&Usart_RxData==0x50) 
	{
		RxState=2;
	}
	else if(RxState==2)
	{	
		id_Flag=Usart_RxData-48;
		RxState=3;
	}
	else if(RxState==3&&Usart_RxData==0x3D) 
	{
		RxState=4;
	}
	else if(RxState==4)
	{	
		if(Usart_RxData==0x21)
		{
			Data_BitNum=pRxPacket;
			pRxPacket=0;
			RxState=0;
			Usart_RxFlag=1;
		}
		else
		{
			Usart_RxPacket[pRxPacket++]=Usart_RxData;
		}
	}
}

uint8_t Get_id_Flag(void)
{
  uint8_t id_temp;
  id_temp=id_Flag;
  id_Flag=0;
  return id_temp;
}

float Pow_invert(uint8_t X,uint8_t n)
{
  float result=X;
	while(n--)
	{
		result/=10;
	}
	return result;
}

float RxPacket_Data_Handle(void)
{
  float Data=0.0;
  uint8_t dot_Flag=0;
  uint8_t dot_after_num=1;
  int8_t minus_Flag=1;
  for(uint8_t i=0;i<Data_BitNum;i++)
  {
    if(Usart_RxPacket[i]==0x2D)
    {
      minus_Flag=-1;
      continue;
    }
    if(dot_Flag==0)
    {
      if(Usart_RxPacket[i]==0x2E)
      {
        dot_Flag=1;
      }
      else
      {
        Data = Data*10 + Usart_RxPacket[i]-48;
      }
    }
    else
    {
      Data = Data + Pow_invert(Usart_RxPacket[i]-48,dot_after_num);
      dot_after_num++;
    }
  }
  return Data*minus_Flag;
}
