#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <pthread.h>
#include <arpa/inet.h>

#define BUF 1024
#define PORT 3500

char system_msg[2][6] = { "@show\n","@exit\n" };

void *rcv_msg(void * arg);
void *send_msg(void * arg);

int main(int argc, char *argv[]) {
	int sock;
	char *addr;
	char flag[2];
	char buf[BUF];
	char name[30];

	struct sockaddr_in serv_addr;
	pthread_t recv_thread, send_thread;

	if (argc != 3) {
		printf("Usage : %s <IP> <ID> \n", argv[0]);
		exit(1);
	}
	strcpy(name, argv[2]);
	sock = socket(PF_INET, SOCK_STREAM, 0);
	if (-1 == sock) {
		printf("SOCKET ERROR\n");
		exit(1);
	}
	memset(&serv_addr, 0, sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = inet_addr(argv[1]);
	serv_addr.sin_port = htons(PORT);

	if (-1 == connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))) {
		printf("CONNECT ERROR\n");
		exit(1);
	}

	// ~ connect() ���� set up

	memset(buf, 0, sizeof(buf));
	read(sock, flag, 2); // ���� ���� flag read
	flag[1] = '\0';
	read(sock, buf, sizeof(buf));
	write(sock, name, sizeof(name)); // �̸� wirte
	if (!strcmp(flag, "1")) {
		printf("%s, %s \n", argv[2], buf);
		memset(buf, 0, sizeof(buf));

		pthread_create(&recv_thread, NULL, rcv_msg, (void *)&sock);
		pthread_create(&send_thread, NULL, send_msg, (void *)&sock);
		//������ �޽��� ó���ϴ� ������ ����

		pthread_join(recv_thread, NULL);
		pthread_join(send_thread, NULL);
	}
	else if (!strcmp(flag, "0")) printf("Server is fulled\n");
	// �ִ� Ŭ���̾�Ʈ �� �Ѿ��� ���
	close(sock);

	return 0;
}
void *rcv_msg(void * arg) {
	//�޼����� �޴� �Լ�
	int sock = *((int *)arg);
	char name_message[BUF];
	memset(name_message, 0, sizeof(name_message));
	int recv_size;
	int rtn;

	while (1) {
		memset(name_message, 0, sizeof(name_message));
		recv_size = read(sock, name_message, BUF);
		if (recv_size == -1)
			break;

		else if (!strcmp(name_message, "@exit\n")) //@exit�� �� ���� ����										   //exit(1);
			break;

		name_message[recv_size] = '\0';
		fputs(name_message, stdout);//name_message���

	}
}
void *send_msg(void * arg) {
	//�޼����� ������ �Լ�
	int sock = *((int*)arg);
	char message[BUF];

	memset(message, 0, sizeof(message));
	while (1) {

		fgets(message, BUF, stdin);

		if (!strcmp(message, "@exit\n")) { //@exit������ ������ �����ְ� ��������
			write(sock, message, strlen(message));
			break;
		}
		else
			write(sock, message, strlen(message));
		memset(message, 0, sizeof(message));

	}
}