/*
write an aligned malloc and free function
which supports allocating memory such that memory is divisible by a specific power of two
*/

void* alloc(size_t byes, size_t align){
	void* orig;
	void** aligned;
	int offset = align-1+sizeof(void*);
	orig = malloc(bytes+offset);
	aligned = (void**)( ((size_t)orig+offset) & (~(align-1)) );
	aligned[-1]=orig;
	return aligned;
}

void free(void* aligned){
	void* orig = ((void**)aligned)[-1];
	free(orig);
}