class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        if not self.head:
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        try:
            if not self.head:
                raise Exception("List is empty.")
            if n <= 0:
                raise Exception("Position must be a positive integer.")

            if n == 1:
                print(f"Deleting node at position {n} with value {self.head.data}")
                self.head = self.head.next
                return

            current = self.head
            prev = None
            count = 1

            while current and count < n:
                prev = current
                current = current.next
                count += 1

            if not current:
                raise Exception("Position out of range.")

            print(f"Deleting node at position {n} with value {current.data}")
            prev.next = current.next

        except Exception as e:
            print("Error:", e)

# ========= USER INTERACTION =========

ll = LinkedList()

# Input list size and elements
try:
    n = int(input("Enter number of elements in the list: "))
    for i in range(n):
        val = int(input(f"Enter element {i+1}: "))
        ll.add_node(val)

    print("\nInitial Linked List:")
    ll.print_list()

    # Delete a node
    pos = int(input("\nEnter the position (1-based) of node to delete: "))
    ll.delete_nth_node(pos)

    print("\nLinked List after deletion:")
    ll.print_list()

except ValueError:
    print("Please enter valid integers.")
