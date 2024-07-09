import numpy as np
import pdb


class VectorStore:
    def __init__(self):
        # dictionary to store vectors
        self.vector_data = {}
        # dictionary to store indexing structure for retrieval
        self.vector_index = {}

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the vector store

        Args:
            vector_id (str or int): a unique id for the vector
            vector (numpy.darray): The vector data to be stored
        """
        self.vector_data[vector_id] = vector
        self.update_index(vector_id, vector)

    def get_vector(self, vector_id):
        """
        Get a vector from the vector store

        Args: 
            vector_id (str or int): A unique id for the vector

        Returns:
            numpy.darray: the vector data if found, or None otherwise
        """
        return self.vector_data.get(vector_id)

    def update_index(self, vector_id, vector):
        """
        Update the indexing structure for the vector store

        Args:
            vector_id (str or int): a unique id for the vector
            vector (numpy.darray): The vector data to be stored
        """
        for existing_id, exisiting_vector in self.vector_data.items():
            similarity = np.dot(vector, exisiting_vector) / \
                (np.linalg.norm(vector) * np.linalg.norm(exisiting_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity

    def find_similar_vectors(self, query_vector, num_results=5):
        """
        Find similar vectors to the query vector

        Args: 
            query_vector (numpy.darray): The query vector
            num_results (int): The number of results to return

        Returns:
            list: A list of tuples of the form (vector_id, similarity_score)
        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(
                query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        # sort the similarities in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        # return the top N results
        return results[:num_results]
